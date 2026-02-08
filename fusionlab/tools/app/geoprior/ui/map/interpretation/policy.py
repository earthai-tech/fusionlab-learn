# geoprior/ui/map/interpretation/policy.py
# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>

"""geoprior.ui.map.interpretation.policy

Policy-ready interpretation helpers.

This is a small, UI-agnostic layer that enriches
hotspot payloads with confidence, actions, and
export helpers.
"""

from __future__ import annotations

import json
import textwrap
from dataclasses import dataclass
from typing import Any, Callable, Dict
from typing import Iterable, List, Optional

from ..keys import ( 
    MAP_VIEW_INTERP_ENABLED, # "map.view.interp.enabled"
    MAP_VIEW_INTERP_SCHEME, # "map.view.interp.scheme"
    MAP_VIEW_INTERP_CALLOUTS, # "map.view.interp.callouts"
    MAP_VIEW_INTERP_CALLOUT_LEVEL, # "map.view.interp.callout_level"
    MAP_VIEW_INTERP_CALLOUT_ACTIONS, # "map.view.interp.callout_actions"
    MAP_VIEW_INTERP_ACTION_PACK, # "map.view.interp.action_pack"
    MAP_VIEW_INTERP_ACTION_INTENSITY, # "map.view.interp.action_intensity"
    MAP_VIEW_INTERP_TONE, # "map.view.interp.tone"
 
    MAP_VIEW_INTERP_CALL_STD_MAX,
    MAP_VIEW_INTERP_CALL_DET_MAX,
    MAP_VIEW_INTERP_CALL_WRAP_COLS,
    MAP_VIEW_INTERP_CALL_POPUP,
)


_CALLOUT_WRAP_COLS = 46
_CALLOUT_MAX_LINES = 6

@dataclass(frozen=True)
class InterpCfg:
    enabled: bool = False
    scheme: str = "subsidence"  # subsidence|risk
    callouts: bool = True

    callout_level: str = "standard"
    callout_actions: bool = True
    call_std_max_lines: int = 4
    call_det_max_lines: int = 10
    call_wrap_cols: int = _CALLOUT_WRAP_COLS
    call_popup: bool = True
    tone: str = "municipal"  # technical|municipal|public

    action_pack: str = "balanced"
    action_intensity: str = "balanced"

    manual_conf: Optional[float] = None


def _wrap_text(
    s: str,
    *,
    width: int = _CALLOUT_WRAP_COLS,
) -> List[str]:
    txt = str(s or "").strip()
    if not txt:
        return []
    return textwrap.wrap(
        txt,
        width=width,
        break_long_words=False,
        break_on_hyphens=False,
    )


def _wrap_join_bits(
    bits: List[str],
    *,
    sep: str = " | ",
    width: int = _CALLOUT_WRAP_COLS,
) -> List[str]:
    """Pack tokens into multiple lines using preferred breakpoints."""
    out: List[str] = []
    cur = ""

    for b in bits:
        tok = str(b or "").strip()
        if not tok:
            continue

        cand = tok if not cur else (cur + sep + tok)
        if len(cand) <= width:
            cur = cand
            continue

        if cur:
            out.append(cur)
        cur = tok

    if cur:
        out.append(cur)

    return out


def _cap_lines(
    lines: List[str],
    *,
    max_lines: int = _CALLOUT_MAX_LINES,
    suffix: str = " …",
) -> List[str]:
    if max_lines <= 0:
        return list(lines or [])
    if len(lines) <= max_lines:
        return lines
    keep = lines[: max_lines]
    keep[-1] = keep[-1].rstrip() + str(suffix)
    return keep

def cfg_from_get(
    getter: Callable[[str, Any], Any],
) -> InterpCfg:
    """Read interpretation config from a store-like getter."""

    def _b(key: str, default: bool) -> bool:
        return bool(getter(key, default))

    def _s(key: str, default: str) -> str:
        v = getter(key, default)
        return str(v or default).strip()
    
    def _i(key: str, default: int) -> int:
        v = getter(key, default)
        try:
            return int(v)
        except :
            return int(default)

    def _clamp_i(v: int, lo: int, hi: int) -> int:
        return max(lo, min(hi, int(v)))


    mc = getter("map.view.interp.confidence", None)
    if mc is not None:
        try:
            mc = float(mc)
        except Exception:
            mc = None

    return InterpCfg(
        enabled=_b(MAP_VIEW_INTERP_ENABLED, False),
        scheme=_s(MAP_VIEW_INTERP_SCHEME, "subsidence"),
        callouts=_b(MAP_VIEW_INTERP_CALLOUTS, True),
        callout_level=_s(
            MAP_VIEW_INTERP_CALLOUT_LEVEL,
            "standard",
        ),
        callout_actions=_b(
            MAP_VIEW_INTERP_CALLOUT_ACTIONS,
            True,
        ),
        call_std_max_lines=_clamp_i(
            _i(MAP_VIEW_INTERP_CALL_STD_MAX, 4),
            1,
            20,
        ),
        call_det_max_lines=_clamp_i(
            _i(MAP_VIEW_INTERP_CALL_DET_MAX, 10),
            0,
            60,
        ),
        call_wrap_cols=_clamp_i(
            _i(
                MAP_VIEW_INTERP_CALL_WRAP_COLS,
                _CALLOUT_WRAP_COLS,
            ),
            24,
            84,
        ),
        call_popup=_b(
            MAP_VIEW_INTERP_CALL_POPUP,
            True,
        ),
        tone=_s(MAP_VIEW_INTERP_TONE, "municipal"),
        action_pack=_s(
            MAP_VIEW_INTERP_ACTION_PACK,
            "balanced",
        ),
        action_intensity=_s(
            MAP_VIEW_INTERP_ACTION_INTENSITY,
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
        trig = _policy_trigger_for(
                sev=sev,
                conf=conf,
                conf_tag=tag,
                n=n,
                scheme=cfg.scheme,
            )
        act = _action_for(
            sev=sev,
            scheme=cfg.scheme,
            pack=cfg.action_pack,
            intensity=cfg.action_intensity,
            tone=cfg.tone,
        )

        steps = _policy_steps_for(
            sev=sev,
            scheme=cfg.scheme,
            pack=cfg.action_pack,
            intensity=cfg.action_intensity,
            tone=cfg.tone,
            action=act,
        )

        stage = _policy_stage_for(
            sev=sev,
            scheme=cfg.scheme,
        )

        owner = _policy_owner_for(
            scheme=cfg.scheme,
            pack=cfg.action_pack,
            stage=stage,
        )
        deadline_days = _policy_deadline_days_for(
            sev=sev,
            scheme=cfg.scheme,
            stage=stage,
        )
        evid = _policy_evidence_for(
            sev=sev,
            n=n,
            conf=conf,
            basis=basis,
            metric=metric,
            unit=unit,
        )
       
        hh["conf"] = float(conf)
        hh["conf_tag"] = tag
        hh["action"] = act
        hh["policy_trigger"] = trig
        hh["policy_steps"] = steps
        hh["policy_stage"] = stage
        hh["policy_owner"] = owner
        hh["policy_deadline_days"] = int(deadline_days)
        hh["policy_evidence"] = evid

        if cfg.enabled and cfg.callouts:
            lab_full = _callout_label(
                hh,
                cfg=cfg,
                basis=basis,
                metric=metric,
                unit=unit, 
                full=True,
            )
            hh["label"] = _callout_label(
                hh,
                cfg=cfg,
                basis=basis,
                metric=metric,
                unit=unit,
                full=False,
             )
            if cfg.call_popup and lab_full:
                hh["label_full"] = lab_full

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
                "policy_trigger": str(
                    h.get("policy_trigger", "")
                    or ""
                ),
                "policy_steps": json.dumps(
                    h.get("policy_steps", []),
                    ensure_ascii=False,
                ),
                "policy_stage": str(
                    h.get("policy_stage", "")
                    or ""
                ),
                "policy_owner": str(
                    h.get("policy_owner", "")
                    or ""
                ),
                "policy_deadline_days": int(
                    h.get("policy_deadline_days", 0)
                    or 0
                ),
                "policy_evidence": str(
                    h.get("policy_evidence", "") or ""
                ),
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

def _policy_evidence_for(
    *,
    sev: str,
    n: int,
    conf: float,
    basis: str,
    metric: str,
    unit: str = "",
) -> str:
    s = str(sev or "low").lower()
    nn = max(0, int(n))
    c = float(conf)

    b = str(basis or "").strip() or "current"
    m = str(metric or "").strip() or "value"
    u = str(unit or "").strip()

    tail = f"{b}/{m}"
    if u:
        tail = f"{tail} ({u})"

    return (
        f"sev={s}; n={nn}; conf={c:.2f}; "
        f"{tail}"
    )

def _policy_trigger_for(
    *,
    sev: str,
    conf: float,
    conf_tag: str,
    n: int,
    scheme: str,
) -> str:
    s = str(sev or "low").lower()
    sc = str(scheme or "subsidence").lower()
    ct = str(conf_tag or "").lower()

    nn = max(0, int(n))

    base = _TRIGGER_BASE.get(s, _TRIGGER_BASE["low"])
    if sc == "risk":
        base = _TRIGGER_RISK.get(s, base)

    if ct == "low" or nn < 10:
        return (
            "Verify first: " + base +
            " (low confidence; "
            "confirm with independent data)."
        )

    if ct == "medium" or nn < 30:
        return (
            base +
            " (moderate confidence; "
            "confirm before major controls)."
        )

    return base


_TRIGGER_BASE: Dict[str, str] = {
    "critical": (
        "Emergency trigger: severe subsidence hotspot; "
        "activate incident response and protect "
        "critical assets."
    ),
    "high": (
        "Response trigger: high subsidence hotspot; "
        "inspect assets, review pumping/loads, and "
        "apply near-term controls."
    ),
    "medium": (
        "Response trigger: emerging subsidence; "
        "validate trend and prepare targeted "
        "mitigation plan."
    ),
    "low": (
        "Watch trigger: baseline deformation; "
        "continue routine monitoring and "
        "track trends."
    ),
}

_TRIGGER_RISK: Dict[str, str] = {
    "critical": (
        "Emergency trigger: critical risk hotspot; "
        "activate safety measures and restrict "
        "high-risk activity."
    ),
    "high": (
        "Response trigger: high risk hotspot; "
        "prioritize vulnerable assets and apply "
        "risk mitigation."
    ),
    "medium": (
        "Response trigger: moderate risk; "
        "update plans and monitor risk indicators."
    ),
    "low": (
        "Watch trigger: low risk; "
        "continue risk tracking."
    ),
}

def _policy_deadline_days_for(
    *,
    sev: str,
    scheme: str,
    stage: str,
) -> int:
    s = str(sev or "low").lower()
    sc = str(scheme or "subsidence").lower()
    st = str(stage or "").lower()

    base = _DEADLINE_DAYS.get(
        s,
        _DEADLINE_DAYS["medium"],
    )

    if st == "emergency":
        base = _EMERGENCY_DEADLINE_DAYS.get(
            s,
            _EMERGENCY_DEADLINE_DAYS["medium"],
        )

    # Optional: slightly tighter for "risk" scheme.
    if sc == "risk":
        base = int(max(1, round(base * 0.7)))

    return int(base)


_DEADLINE_DAYS: Dict[str, int] = {
    "critical": 7,
    "high": 30,
    "medium": 90,
    "low": 180,
}

_EMERGENCY_DEADLINE_DAYS: Dict[str, int] = {
    "critical": 3,
    "high": 7,
    "medium": 14,
    "low": 30,
}

def _policy_owner_for(
    *,
    scheme: str,
    pack: str,
    stage: str,
) -> str:
    sc = str(scheme or "subsidence").lower()
    pk = str(pack or "balanced").lower()
    st = str(stage or "").lower()

    base = _POLICY_OWNER.get(
        sc,
        _POLICY_OWNER["subsidence"],
    )

    # Emergency stage can override owner.
    if st == "emergency":
        return base.get(
            "__emergency__",
            "Emergency Management",
        )

    return base.get(
        pk,
        base.get("balanced", "Municipal Taskforce"),
    )


_POLICY_OWNER: Dict[str, Dict[str, str]] = {
    "subsidence": {
        "__emergency__": (
            "Emergency Mgmt + Public Works"
        ),
        "balanced": "Municipal Taskforce",
        "groundwater": (
            "Water Resources Department"
        ),
        "infrastructure": (
            "Public Works / Utilities"
        ),
        "planning": "Urban Planning Department",
        "monitoring": (
            "Geospatial Monitoring Unit"
        ),
    },
    "risk": {
        "__emergency__": (
            "Emergency Management (EOC)"
        ),
        "balanced": "Risk Management Office",
        "groundwater": (
            "Water Resources Department"
        ),
        "infrastructure": (
            "Public Works / Utilities"
        ),
        "planning": "Urban Planning Department",
        "monitoring": (
            "Monitoring + Risk Analytics Unit"
        ),
    },
}

def _policy_stage_for(
    *,
    sev: str,
    scheme: str,
) -> str:
    # Minimal, decisive staging for policy response.
    # watch -> response -> emergency
    sc = str(scheme or "subsidence").lower()
    s = str(sev or "low").lower()

    base = _POLICY_STAGE.get(
        sc,
        _POLICY_STAGE["subsidence"],
    )
    return str(base.get(s, "watch"))


_POLICY_STAGE: Dict[str, Dict[str, str]] = {
    "subsidence": {
        "critical": "emergency",
        "high": "response",
        "medium": "watch",
        "low": "watch",
    },
    "risk": {
        "critical": "emergency",
        "high": "response",
        "medium": "watch",
        "low": "watch",
    },
}

def _policy_steps_for(
    *,
    sev: str,
    scheme: str,
    pack: str,
    intensity: str,
    tone: str,
    action: str,
) -> List[str]:
    sc = str(scheme or "subsidence").lower()
    s = str(sev or "low").lower()
    p = str(pack or "balanced").lower()

    base = _POLICY_STEPS.get(sc, _POLICY_STEPS["subsidence"])
    row = base.get(s, base.get("low", {}))

    steps = row.get(p) or row.get("balanced") or []
    if not steps:
        steps = _steps_from_action(action)

    out: List[str] = []
    for st in steps:
        msg = str(st or "").strip()
        if not msg:
            continue

        if str(intensity or "").lower() == "conservative":
            msg = _soften(msg)
        elif str(intensity or "").lower() == "aggressive":
            msg = _strengthen(msg)

        if str(tone or "").lower() == "public":
            msg = _public(msg)

        out.append(msg)

    return out[:6]


def _steps_from_action(action: str) -> List[str]:
    txt = str(action or "").strip()
    if not txt:
        return []

    # Split on semicolons first (your messages use them)
    parts = [p.strip() for p in txt.split(";")]
    parts = [p for p in parts if p]

    out: List[str] = []
    for p in parts:
        # Keep short, remove trailing dot.
        s = p.strip().rstrip(".")
        if not s:
            continue
        # Capitalize first letter lightly.
        out.append(s[0:1].upper() + s[1:])

    return out[:6]

_POLICY_STEPS: Dict[str, Dict[str, Dict[str, List[str]]]] = {
    "subsidence": {
        "critical": {
            "balanced": [
                "Activate hotspot response (Level 1).",
                "Dispatch field check within 72 hours.",
                "Inspect roads, pipes, and foundations.",
                "Cap pumping in a 1–3 km buffer zone.",
                "Run weekly deformation checks.",
                "Publish advisory + escalation triggers.",
            ],
            "groundwater": [
                "Freeze new well permits in buffer.",
                "Audit wells; enforce metering + logs.",
                "Cap pumping now; plan staged cuts.",
                "Check drawdown drivers and leakage.",
                "Start managed recharge if feasible.",
                "Issue compliance notice + follow-up.",
            ],
            "infrastructure": [
                "Inspect bridges, roads, water mains.",
                "Pressure-test mains; fix major leaks.",
                "Check sewer lines and manholes.",
                "Restrict heavy loads on weak assets.",
                "Install settlement markers on assets.",
                "Prioritize emergency maintenance list.",
            ],
            "planning": [
                "Pause deep excavation in buffer.",
                "Require geotech + dewatering plan.",
                "Apply temporary load limits by permit.",
                "Update zoning constraints for buffer.",
                "Prioritize resilient retrofit program.",
                "Set review board for exceptions.",
            ],
            "monitoring": [
                "Deploy GNSS benchmarks + leveling.",
                "Add piezometers for groundwater.",
                "Set alert thresholds and triggers.",
                "Increase cadence to weekly/biweekly.",
                "Validate QA; document sensor health.",
                "Publish monthly hotspot bulletin.",
            ],
        },
        "high": {
            "balanced": [
                "Schedule targeted inspection this month.",
                "Tighten pumping controls in buffer.",
                "Increase monitoring cadence to biweekly.",
                "Set escalation thresholds and owners.",
                "Publish a short risk note for permits.",
            ],
            "groundwater": [
                "Review permits and meter compliance.",
                "Reduce extraction in buffer (staged).",
                "Compare pumping vs groundwater trends.",
                "Prepare alternative supply options.",
                "Plan recharge feasibility screening.",
            ],
            "infrastructure": [
                "Inspect critical assets in buffer.",
                "Prioritize leakage repair program.",
                "Add settlement markers on key lines.",
                "Update asset risk map monthly.",
                "Prepare contingency maintenance plan.",
            ],
            "planning": [
                "Add constraints in permit workflow.",
                "Limit basement depth and dewatering.",
                "Require settlement-risk screening.",
                "Flag buffer zone in zoning layers.",
                "Review exceptions case-by-case.",
            ],
            "monitoring": [
                "Increase monitoring frequency now.",
                "Cross-check satellite vs ground data.",
                "Add early-warning thresholds.",
                "Maintain monthly hotspot bulletin.",
                "Validate data QA and metadata.",
            ],
        },
        "medium": {
            "balanced": [
                "Maintain routine monitoring schedule.",
                "Track trends and update quarterly.",
                "Prepare escalation playbook if needed.",
                "Integrate risk into permit screening.",
            ],
            "groundwater": [
                "Track extraction and groundwater.",
                "Identify peak demand stress periods.",
                "Improve reporting coverage.",
                "Prepare mitigation options if worse.",
            ],
            "infrastructure": [
                "Maintain routine inspections.",
                "Track maintenance and leak reports.",
                "Flag assets for follow-up checks.",
                "Review exposure quarterly.",
            ],
            "planning": [
                "Flag buffer as watch area.",
                "Require basic settlement screening.",
                "Review permits quarterly.",
                "Keep monitoring and trend notes.",
            ],
            "monitoring": [
                "Baseline monitoring with QA checks.",
                "Validate against independent data.",
                "Publish quarterly summaries.",
                "Confirm sensor health regularly.",
            ],
        },
        "low": {
            "balanced": [
                "Baseline monitoring and record keeping.",
                "Reassess if trend accelerates.",
                "Check again after major new pumping.",
            ],
        },
    },
    "risk": {
        "critical": {
            "balanced": [
                "Activate safety protocol immediately.",
                "Protect critical facilities first.",
                "Restrict high-risk activity in buffer.",
                "Publish public guidance and contacts.",
                "Monitor closely with clear triggers.",
            ],
        },
        "high": {
            "balanced": [
                "Prioritize vulnerable assets.",
                "Apply preventive restrictions as needed.",
                "Monitor and reassess monthly.",
                "Update risk map for decision makers.",
            ],
        },
        "medium": {
            "balanced": [
                "Track risk indicators routinely.",
                "Integrate risk into planning process.",
                "Review exposure quarterly.",
            ],
        },
        "low": {
            "balanced": [
                "Keep baseline risk tracking.",
                "Reassess after new development.",
            ],
        },
    },
}

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


def _wrap_callout(
    bits: List[str],
    *,
    sep: str = " | ",
    width: int = 52,
) -> str:
    lines: List[str] = []
    cur = ""

    for b in (bits or []):
        s = str(b or "").strip()
        if not s:
            continue

        if not cur:
            cur = s
            continue

        cand = f"{cur}{sep}{s}"
        if len(cand) <= width:
            cur = cand
            continue

        lines.append(cur)
        cur = s

    if cur:
        lines.append(cur)

    out: List[str] = []
    for ln in lines:
        if len(ln) <= width:
            out.append(ln)
            continue

        out.extend(
            textwrap.wrap(
                ln,
                width=width,
                break_long_words=False,
                break_on_hyphens=False,
            )
        )

    return "\n".join(out)

def _callout_label(
    h: Dict[str, Any],
    *,
    cfg: InterpCfg,
    basis: str,
    metric: str,
    unit: str = "",
    full: bool = False,
) -> str:

    sev = str(h.get("sev", "")) or "low"
    rk = int(h.get("rank", 0) or 0)
    n = int(h.get("n", 0) or 0)
    u = f" {unit}" if unit else ""
    conf = str(h.get("conf_tag", ""))

    lvl = str(cfg.callout_level or "standard").lower()
    wrap = int(getattr(cfg, "call_wrap_cols", _CALLOUT_WRAP_COLS))
    width = max(40, min(84, int(wrap) + 8))

    if lvl == "compact":
        return f"{sev} · #{rk}"

    act = str(h.get("action", ""))
    stage = str(h.get("policy_stage", "") or "").strip()
    owner = str(h.get("policy_owner", "") or "").strip()

    dd = h.get("policy_deadline_days", None)
    try:
        d0 = int(dd) if dd is not None else 0
    except Exception:
        d0 = 0
    due = f"due≤{d0}d" if d0 > 0 else ""

    def _suffix() -> str:
        if getattr(cfg, "call_popup", True):
            return " … (click)"
        return " …"

    def _wrap_bits(bits: List[str]) -> List[str]:
        s = _wrap_callout(bits, width=width)
        return [x for x in s.splitlines() if x.strip()]

    if lvl == "detailed":
        v = _fmt_float(h.get("v", None))
        m = str(metric or "")
        b = str(basis or "")

        base_bits = [
            f"{sev} · #{rk}",
            f"n={n} · conf={conf}",
            f"v={v}{u} · {b} · {m}",
        ]

        pol_bits: List[str] = []
        if stage:
            pol_bits.append(f"stage={stage}")
        if owner:
            pol_bits.append(f"owner={owner}")
        if due:
            pol_bits.append(due)
        if pol_bits:
            base_bits.append(" · ".join(pol_bits))

        lines: List[str] = []
        for ln in base_bits:
            lines.extend(_wrap_text(ln, width=wrap))

        if cfg.callout_actions and act:
            lines.extend(_wrap_text(act, width=wrap))

        if full:
            return "\n".join(lines)

        max_lines = int(getattr(cfg, "call_det_max_lines", 10))
        if max_lines > 0 and len(lines) > max_lines:
            if cfg.callout_actions and act:
                lines2: List[str] = []
                for ln in base_bits:
                    lines2.extend(_wrap_text(ln, width=wrap))
                if len(lines2) <= max_lines:
                    return "\n".join(lines2)
                lines = lines2

            lines = _cap_lines(
                lines,
                max_lines=max_lines,
                suffix=_suffix(),
            )
        return "\n".join(lines)

    head = f"{sev} · #{rk} · n={n} · {conf}"
    bits: List[str] = [head]

    opt: List[str] = []
    if stage:
        opt.append(f"stage={stage}")
    if owner:
        opt.append(f"owner={owner}")
    if due:
        opt.append(due)
    if cfg.callout_actions and act:
        opt.append(act)

    if full:
        return "\n".join(_wrap_bits(bits + opt))

    max_lines = int(getattr(cfg, "call_std_max_lines", 4))
    cand = bits + opt
    lines = _wrap_bits(cand)

    if len(lines) > max_lines and opt:
        if cfg.callout_actions and act and act in opt:
            opt2 = [x for x in opt if x != act]
            lines2 = _wrap_bits(bits + opt2)
            if len(lines2) <= max_lines:
                return "\n".join(lines2)
            opt = opt2
            lines = lines2

    if len(lines) > max_lines and owner:
        tgt = f"owner={owner}"
        if tgt in opt:
            opt2 = [x for x in opt if x != tgt]
            lines2 = _wrap_bits(bits + opt2)
            if len(lines2) <= max_lines:
                return "\n".join(lines2)
            opt = opt2
            lines = lines2

    if len(lines) > max_lines and stage:
        tgt = f"stage={stage}"
        if tgt in opt:
            opt2 = [x for x in opt if x != tgt]
            lines2 = _wrap_bits(bits + opt2)
            if len(lines2) <= max_lines:
                return "\n".join(lines2)
            opt = opt2
            lines = lines2

    if len(lines) > max_lines and due and due in opt:
        opt2 = [x for x in opt if x != due]
        lines2 = _wrap_bits(bits + opt2)
        if len(lines2) <= max_lines:
            return "\n".join(lines2)
        opt = opt2
        lines = lines2

    if len(lines) > max_lines:
        lines = _cap_lines(
            lines,
            max_lines=max_lines,
            suffix=_suffix(),
        )

    return "\n".join(lines)

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
    m = msg
    m = m.replace("Freeze", "Pause")
    m = m.replace("freeze", "pause")
    m = m.replace("moratorium", "temporary pause")
    m = m.replace("cap", "review and cap")
    m = m.replace("restrict", "review and restrict")
    return m


def _strengthen(msg: str) -> str:
    m = msg
    m = m.replace("review", "enforce")
    m = m.replace("tighten", "enforce")
    m = m.replace("prepare", "initiate")
    m = m.replace("increase", "increase immediately")
    return m


def _public(msg: str) -> str:
    m = msg
    m = m.replace("InSAR/GNSS/leveling", "satellite and ground checks")
    m = m.replace("piezometers", "groundwater sensors")
    m = m.replace("moratorium", "temporary stop")
    m = m.replace("buffer", "nearby area")
    m = m.replace("monitoring", "tracking")
    return m

_ACTION: Dict[str, Dict[str, Dict[str, str]]] = {
    "subsidence": {
        "critical": {
            "balanced": (
                "Declare hotspot response (Level 1); dispatch "
                "field team within 72h; inspect roads/pipes/"
                "foundations; set a temporary pumping cap in a "
                "1–3 km buffer; start weekly deformation checks "
                "(InSAR/GNSS/leveling); publish a public advisory."
            ),
            "groundwater": (
                "Freeze new well permits in buffer; enforce "
                "metering + reporting; audit active wells; cap "
                "pumping immediately and schedule staged "
                "reductions; review drawdown drivers; launch a "
                "recovery plan (managed recharge where feasible)."
            ),
            "infrastructure": (
                "Emergency inspection of critical assets; "
                "pressure-test water mains; check sewer lines "
                "and manholes; restrict heavy loads on affected "
                "corridors; repair leaks; install settlement "
                "markers on priority structures."
            ),
            "planning": (
                "Temporary moratorium on deep excavation and "
                "high-load projects in buffer; require geotech "
                "review + dewatering plan for any exception; "
                "update zoning constraints; prioritize resilient "
                "retrofits in the hotspot corridor."
            ),
            "monitoring": (
                "Deploy GNSS benchmarks + leveling lines; add "
                "piezometers; set alert thresholds and escalation "
                "rules; increase measurement cadence (weekly to "
                "biweekly); validate instruments and data QA."
            ),
        },
        "high": {
            "balanced": (
                "Targeted inspection this month; tighten pumping "
                "controls in buffer; verify meter compliance; "
                "increase monitoring cadence (monthly→biweekly); "
                "define trigger thresholds for escalation."
            ),
            "groundwater": (
                "Permit review + compliance checks; reduce "
                "extraction in buffer (staged); enforce reporting; "
                "compare pumping vs. groundwater trends; prepare "
                "recharge/alternative supply options."
            ),
            "infrastructure": (
                "Inspect critical assets; prioritize leakage "
                "repairs; schedule maintenance; update asset risk "
                "map; add settlement markers on key lines/roads."
            ),
            "planning": (
                "Add planning constraints in buffer; require "
                "geotech and settlement-risk note for new builds; "
                "limit basement depth/dewatering; integrate risk "
                "into permitting checklist."
            ),
            "monitoring": (
                "Increase monitoring; add early-warning thresholds; "
                "cross-check satellite vs. ground data; keep a "
                "monthly hotspot bulletin."
            ),
        },
        "medium": {
            "balanced": (
                "Routine monitoring; integrate into planning; "
                "prioritize targeted checks; track trends and "
                "prepare an escalation playbook."
            ),
            "groundwater": (
                "Track extraction + groundwater; identify peak "
                "demand periods; improve reporting coverage; "
                "prepare mitigation options if trends worsen."
            ),
            "infrastructure": (
                "Routine inspection schedule; track maintenance "
                "records; verify drainage and utility integrity; "
                "flag assets for follow-up if deformation rises."
            ),
            "planning": (
                "Flag area for future constraints; require basic "
                "settlement-risk screening; keep monitoring and "
                "review permits quarterly."
            ),
            "monitoring": (
                "Baseline monitoring; validate with independent "
                "data; keep quarterly summaries; confirm sensor "
                "health and metadata."
            ),
        },
        "low": {
            "balanced": (
                "Baseline monitoring; keep records; revisit if "
                "trend accelerates or new stressors appear."
            ),
        },
    },
    "risk": {
        "critical": {
            "balanced": (
                "Activate safety protocol; protect critical "
                "facilities (hospitals, pipelines, bridges); "
                "restrict high-risk activity in buffer; define "
                "public safety messaging; monitor closely with "
                "clear escalation triggers."
            ),
        },
        "high": {
            "balanced": (
                "Targeted risk mitigation; prioritize vulnerable "
                "assets; apply preventive restrictions where "
                "needed; monitor and reassess monthly."
            ),
        },
        "medium": {
            "balanced": (
                "Risk tracking; integrate into urban planning; "
                "maintain monitoring; review exposure quarterly."
            ),
        },
        "low": {
            "balanced": "Risk tracking; keep baseline monitoring.",
        },
    },
}
