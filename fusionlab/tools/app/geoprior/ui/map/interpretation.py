# geoprior/ui/map/interpretation.py
# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>

"""geoprior.ui.map.interpretation

Policy-ready interpretation helpers.

This is a small, UI-agnostic layer that enriches
hotspot payloads with confidence, actions, and
export helpers.
"""

from __future__ import annotations

import json

from dataclasses import dataclass
from typing import Any, Callable, Dict
from typing import Iterable, List, Optional


@dataclass(frozen=True)
class InterpCfg:
    enabled: bool = False
    scheme: str = "subsidence"  # subsidence|risk
    callouts: bool = True

    callout_level: str = "standard"
    callout_actions: bool = True
    tone: str = "municipal"  # technical|municipal|public

    action_pack: str = "balanced"
    action_intensity: str = "balanced"

    manual_conf: Optional[float] = None


def cfg_from_get(
    getter: Callable[[str, Any], Any],
) -> InterpCfg:
    """Read interpretation config from a store-like getter."""

    def _b(key: str, default: bool) -> bool:
        return bool(getter(key, default))

    def _s(key: str, default: str) -> str:
        v = getter(key, default)
        return str(v or default).strip()

    mc = getter("map.view.interp.confidence", None)
    if mc is not None:
        try:
            mc = float(mc)
        except Exception:
            mc = None

    return InterpCfg(
        enabled=_b("map.view.interp.enabled", False),
        scheme=_s("map.view.interp.scheme", "subsidence"),
        callouts=_b("map.view.interp.callouts", True),
        callout_level=_s(
            "map.view.interp.callout_level",
            "standard",
        ),
        callout_actions=_b(
            "map.view.interp.callout_actions",
            True,
        ),
        tone=_s("map.view.interp.tone", "municipal"),
        action_pack=_s(
            "map.view.interp.action_pack",
            "balanced",
        ),
        action_intensity=_s(
            "map.view.interp.action_intensity",
            "balanced",
        ),
        manual_conf=mc,
    )


def apply_interp(
    payload: List[Dict[str, Any]],
    *,
    cfg: InterpCfg,
    basis: str,
    metric: str,
    unit : str =""
) -> List[Dict[str, Any]]:
    """Enrich hotspot payload with interpretation fields."""
    if not payload:
        return []

    out: List[Dict[str, Any]] = []
    for h in payload:
        hh = dict(h or {})
        sev = str(hh.get("sev", "")) or "low"
        n = int(hh.get("n", 0) or 0)

        conf = _confidence(n=n, manual=cfg.manual_conf)
        tag = _conf_tag(conf)
        act = _action_for(
            sev=sev,
            scheme=cfg.scheme,
            pack=cfg.action_pack,
            intensity=cfg.action_intensity,
            tone=cfg.tone,
        )

        hh["conf"] = float(conf)
        hh["conf_tag"] = tag
        hh["action"] = act

        if cfg.enabled and cfg.callouts:
            hh["label"] = _callout_label(
                hh,
                cfg=cfg,
                basis=basis,
                metric=metric,
                unit=unit, 
            )

        out.append(hh)

    return out


def rows_for_export(
    payload: List[Dict[str, Any]],
    *,
    basis: str,
    metric: str,
    unit: str=""
) -> List[Dict[str, Any]]:
    """Convert payload into flat rows for CSV export."""
    rows: List[Dict[str, Any]] = []
    for h in payload or []:
        rows.append(
            {
                "id": int(h.get("rank", 0) or 0),
                "lon": float(h.get("lon", 0.0) or 0.0),
                "lat": float(h.get("lat", 0.0) or 0.0),
                "sev": str(h.get("sev", "")),
                "value": float(h.get("v", 0.0) or 0.0),
                "score": float(h.get("score", 0.0) or 0.0),
                "n": int(h.get("n", 0) or 0),
                "conf": float(h.get("conf", 0.0) or 0.0),
                "conf_tag": str(h.get("conf_tag", "")),
                "action": str(h.get("action", "")),
                "basis": str(basis),
                "metric": str(metric),
                "unit": unit
            }
        )
    return rows


def geojson_from_rows(
    rows: Iterable[Dict[str, Any]],
) -> Dict[str, Any]:
    """Build a GeoJSON FeatureCollection from rows."""
    feats: List[Dict[str, Any]] = []
    for r in rows or []:
        lon = float(r.get("lon", 0.0) or 0.0)
        lat = float(r.get("lat", 0.0) or 0.0)
        props = dict(r)
        props.pop("lon", None)
        props.pop("lat", None)
        feats.append(
            {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [lon, lat],
                },
                "properties": props,
            }
        )

    return {
        "type": "FeatureCollection",
        "features": feats,
    }


def dump_geojson(obj: Dict[str, Any]) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2)


def policy_brief_md(
    rows: List[Dict[str, Any]],
    *,
    cfg: InterpCfg,
    ctx: Optional[Dict[str, Any]] = None,
) -> str:
    """Generate a short, policy-friendly Markdown brief."""

    ctx = dict(ctx or {})
    basis = str(ctx.get("basis", "current"))
    metric = str(ctx.get("metric", "high"))
    
    unit = ""
    if ctx:
        unit = str(ctx.get("unit", "") or "").strip()
    
    n = len(rows or [])
    worst = _worst_sev(rows)

    lines: List[str] = []
    lines.append("# Subsidence hotspot brief")
    lines.append("")
    lines.append(f"- Hotspots detected: **{n}**")
    lines.append(f"- Worst severity: **{worst}**")
    lines.append(f"- Basis: **{basis}**")
    lines.append(f"- Metric: **{metric}**")
    lines.append(f"- Scheme: **{cfg.scheme}**")
    if unit:
        lines.append(f"- Unit: **{unit}**")
    lines.append("")

    if not rows:
        lines.append("No hotspots are available.")
        lines.append("")
        return "\n".join(lines)

    lines.append("## Top hotspots")
    lines.append("")

    top = sorted(
        rows,
        key=lambda r: int(r.get("id", 0) or 0),
    )[: min(5, n)]

    for r in top:
        hid = int(r.get("id", 0) or 0)
        sev = str(r.get("sev", ""))
        lon = float(r.get("lon", 0.0) or 0.0)
        lat = float(r.get("lat", 0.0) or 0.0)
        conf = str(r.get("conf_tag", ""))
        act = str(r.get("action", ""))

        loc = f"({lon:.5f}, {lat:.5f})"
        lines.append(f"- **#{hid} {sev}** {loc}")
        lines.append(f"  - Confidence: **{conf}**")
        if act:
            lines.append(f"  - Action: {act}")

    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append(
        "This summary is automated and should be "
        "validated with local data and field checks."
    )
    lines.append("")

    return "\n".join(lines)


# -------------------------------------------------
# Internals
# -------------------------------------------------

def _confidence(*, n: int, manual: Optional[float]) -> float:
    if manual is not None:
        try:
            x = float(manual)
        except Exception:
            x = None
        if x is not None:
            return float(max(0.0, min(1.0, x)))

    nn = max(0, int(n))
    if nn >= 250:
        return 0.92
    if nn >= 120:
        return 0.85
    if nn >= 60:
        return 0.75
    if nn >= 30:
        return 0.60
    if nn >= 10:
        return 0.45
    return 0.30


def _conf_tag(conf: float) -> str:
    c = float(conf)
    if c >= 0.80:
        return "high"
    if c >= 0.60:
        return "medium"
    return "low"


def _action_for(
    *,
    sev: str,
    scheme: str,
    pack: str,
    intensity: str,
    tone: str,
) -> str:
    s = str(sev or "low").lower()
    sc = str(scheme or "subsidence").lower()

    if sc not in ("subsidence", "risk"):
        sc = "subsidence"

    p = str(pack or "balanced").lower()
    it = str(intensity or "balanced").lower()
    tn = str(tone or "municipal").lower()

    base = _ACTION.get(sc, _ACTION["subsidence"])
    row = base.get(s, base.get("low", {}))

    # Choose a pack if present in dict.
    msg = row.get(p) or row.get("balanced") or ""

    if it == "conservative":
        msg = _soften(msg)
    elif it == "aggressive":
        msg = _strengthen(msg)

    if tn == "technical":
        return msg
    if tn == "public":
        return _public(msg)
    return msg


def _callout_label(
    h: Dict[str, Any],
    *,
    cfg: InterpCfg,
    basis: str,
    metric: str,
    unit: str=""
) -> str:
    sev = str(h.get("sev", "")) or "low"
    rk = int(h.get("rank", 0) or 0)
    n = int(h.get("n", 0) or 0)
    u = f" {unit}" if unit else ""
    
    conf = str(h.get("conf_tag", ""))

    lvl = str(cfg.callout_level or "standard").lower()

    if lvl == "compact":
        return f"{sev} · #{rk}"

    act = str(h.get("action", ""))

    if lvl == "detailed":
        v = _fmt_float(h.get("v", None))
        m = str(metric or "")
        b = str(basis or "")
        bits = [
            f"{sev} · #{rk}",
            f"n={n} · conf={conf}",
            f"v={v}{u} · {b} · {m}",
        ]
        if cfg.callout_actions and act:
            bits.append(act)
        return "\n".join(bits)

    # standard
    bits = [f"{sev} · #{rk} · n={n} · {conf}"]
    if cfg.callout_actions and act:
        bits.append(act)
    return " | ".join(bits)


def _fmt_float(x: Any) -> str:
    try:
        v = float(x)
    except Exception:
        return "na"
    return f"{v:.4g}"


def _worst_sev(rows: List[Dict[str, Any]]) -> str:
    order = {
        "critical": 3,
        "high": 2,
        "medium": 1,
        "low": 0,
    }
    best = "low"
    best_s = -1
    for r in rows or []:
        s = str(r.get("sev", "low")).lower()
        sc = order.get(s, -1)
        if sc > best_s:
            best_s = sc
            best = s
    return best


def _soften(msg: str) -> str:
    return msg.replace("restrict", "review")


def _strengthen(msg: str) -> str:
    return msg.replace("review", "restrict")


def _public(msg: str) -> str:
    # Keep it short and less technical.
    return msg.replace("monitoring", "tracking")


_ACTION: Dict[str, Dict[str, Dict[str, str]]] = {
    "subsidence": {
        "critical": {
            "balanced": (
                "Immediate field check; inspect "
                "infrastructure; review pumping "
                "near hotspot; start monitoring."
            ),
            "groundwater": (
                "Immediate pumping review; restrict "
                "new permits nearby; start monitoring."
            ),
            "infrastructure": (
                "Immediate inspection of roads, "
                "pipes, and foundations; deploy "
                "monitoring."
            ),
            "planning": (
                "Temporary planning hold for new "
                "loads; review zoning constraints."
            ),
            "monitoring": (
                "Deploy sensors; increase "
                "measurement frequency; validate data."
            ),
        },
        "high": {
            "balanced": (
                "Targeted inspection; tighten pumping "
                "controls; increase monitoring."
            ),
            "groundwater": (
                "Permit review; reduce extraction "
                "in hotspot buffer; monitor trends."
            ),
            "infrastructure": (
                "Inspect critical assets; schedule "
                "maintenance; monitor deformation."
            ),
            "planning": (
                "Add planning constraints; require "
                "geotech review for new projects."
            ),
            "monitoring": (
                "Increase monitoring; add early "
                "warning thresholds."
            ),
        },
        "medium": {
            "balanced": (
                "Routine monitoring; integrate into "
                "planning; prioritize checks."
            ),
            "groundwater": (
                "Monitor extraction and groundwater; "
                "prepare mitigation options."
            ),
            "infrastructure": (
                "Routine inspection schedule; track "
                "maintenance records."
            ),
            "planning": (
                "Flag area for future constraints; "
                "keep monitoring."
            ),
            "monitoring": (
                "Baseline monitoring; validate with "
                "independent data."
            ),
        },
        "low": {
            "balanced": "Baseline monitoring.",
        },
    },
    "risk": {
        "critical": {
            "balanced": (
                "Immediate safety review; protect "
                "critical facilities; restrict high "
                "risk activity; monitor closely."
            ),
        },
        "high": {
            "balanced": (
                "Targeted risk mitigation; prioritize "
                "vulnerable assets; monitor."
            ),
        },
        "medium": {
            "balanced": (
                "Risk tracking; integrate into urban "
                "planning; monitor."
            ),
        },
        "low": {
            "balanced": "Risk tracking.",
        },
    },
}
