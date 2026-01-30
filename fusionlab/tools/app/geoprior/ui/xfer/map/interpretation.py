# geoprior/ui/xfer/map/interpretation.py
# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>

"""
Interpretation policies for cross-city transfer maps.

GUI-friendly:
- pure python (no TF / sklearn)
- returns structured payload + HTML render
- uses run metrics + map signals (Δ-hotspots, overlap)

Line length target: <= 62 chars (black config).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


__all__ = [
    "InterpretCfg",
    "interpret_transfer",
    "render_html",
    "map_help_html",
    "render_tip",
]


def _f(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def _i(x: Any) -> Optional[int]:
    try:
        if x is None:
            return None
        return int(x)
    except Exception:
        return None


def _b(x: Any) -> bool:
    return bool(x) is True


def _get(d: Dict[str, Any], k: str, dv: Any = None) -> Any:
    try:
        return d.get(k, dv)
    except Exception:
        return dv


def _uniq(items: List[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for s in items:
        s2 = str(s or "").strip()
        if not s2:
            continue
        if s2 in seen:
            continue
        seen.add(s2)
        out.append(s2)
    return out


def _status(sev: int) -> str:
    if sev >= 5:
        return "fail"
    if sev >= 4:
        return "warn"
    if sev >= 2:
        return "info"
    return "ok"


@dataclass(frozen=True)
class InterpretCfg:
    """
    Thresholds for interpretation.

    Notes
    -----
    - map_sig is optional; controller can pass
      Δ-hotspots + overlap signals when available.
    """

    target_coverage: float = 0.80

    # R2 qualitative bands
    r2_good: float = 0.60
    r2_ok: float = 0.30

    # Relative degradation (vs baseline)
    mae_ratio_ok: float = 1.20
    mae_ratio_bad: float = 1.50

    # Coverage bands
    cov_low: float = 0.60
    cov_ok: float = 0.75
    cov_high: float = 0.90

    # Sharpness relative (vs baseline)
    sharp_ratio_bad: float = 1.50

    # Map signal thresholds (coarse defaults)
    dhot_many: int = 6
    dhot_abs_big: float = 1.0
    overlap_low: float = 0.15
    buf_low_n: int = 25


Policy = Dict[str, Any]
Policies = Dict[str, Policy]


POLICIES: Policies = {
    "missing_y_true": {
        "sev": 2,
        "badge": "NO_Y_TRUE",
        "head": "No targets for metrics.",
        "sum": "Test split often lacks y_true.",
        "act": [
            "Evaluate on val split.",
            "Enable test->val fallback if needed.",
        ],
        "opt": [],
    },
    "order_mismatch": {
        "sev": 4,
        "badge": "ORDER_MISMATCH",
        "head": "Feature order mismatch detected.",
        "sum": (
            "Same names but different order can break "
            "transfer semantics."
        ),
        "act": [
            "Fix Stage-1 feature ordering across cities.",
            "Or rerun with allow_reorder_* enabled.",
        ],
        "opt": [],
    },
    "static_shift": {
        "sev": 2,
        "badge": "STATIC_SHIFT",
        "head": "Static schema differs across cities.",
        "sum": (
            "Missing static fields are often filled with "
            "defaults (shift risk)."
        ),
        "act": [
            "Harmonize static feature lists across cities.",
            "Prefer shared, stable static predictors.",
        ],
        "opt": [],
    },
    "weak_baseline": {
        "sev": 5,
        "badge": "WEAK_BASELINE",
        "head": "Baseline performance is weak.",
        "sum": (
            "Fix within-city before transfer decisions."
        ),
        "act": [
            "Tune baseline per city on val split.",
            "Verify scaling, target mapping, and units.",
            "Audit physics loss weights (if PINN).",
        ],
        "opt": [
            "Use Points=All points to inspect coverage.",
        ],
    },
    "low_r2": {
        "sev": 3,
        "badge": "LOW_R2",
        "head": "Low explanatory power (R²).",
        "sum": "Predicted shape does not track targets.",
        "act": [
            "Verify schema match and ordering.",
            "Compare rescale modes (as_is vs strict).",
            "Inspect per-horizon degradation pattern.",
        ],
        "opt": [
            "Enable Links to see hotspot relations.",
        ],
    },
    "undercoverage": {
        "sev": 4,
        "badge": "UNDERCOV",
        "head": "Intervals under-cover.",
        "sum": "Uncertainty is too narrow for target.",
        "act": [
            "Use calib_mode='target' on target val.",
            "Check quantiles include 0.1/0.9.",
            "Avoid mixing scaling spaces for quantiles.",
        ],
        "opt": [],
    },
    "overwide": {
        "sev": 2,
        "badge": "OVERWIDE",
        "head": "Intervals are overly wide.",
        "sum": "Coverage is high but sharpness is poor.",
        "act": [
            "Prefer calib_mode='source' or none.",
            "Cap calibrator factors if supported.",
        ],
        "opt": [],
    },
    "transfer_drop": {
        "sev": 4,
        "badge": "TRANSFER_DROP",
        "head": "Transfer degrades vs baseline.",
        "sum": "Cross-city generalization is limited.",
        "act": [
            "Try warm-start on target (strategy=warm).",
            "Add domain-invariant features / priors.",
            "Use target calibration for evaluation.",
        ],
        "opt": [
            "Points=Hotspots + base for quick scan.",
        ],
    },
    "warm_not_helping": {
        "sev": 3,
        "badge": "WARM_NOT_HELPING",
        "head": "Warm-start does not improve transfer.",
        "sum": "Mismatch may be structural or unstable.",
        "act": [
            "Increase warm epochs (e.g., 5..10).",
            "Lower warm LR (e.g., 1e-5..5e-5).",
            "Freeze backbone, train head only.",
        ],
        "opt": [],
    },
    "map_shift": {
        "sev": 4,
        "badge": "MAP_SHIFT",
        "head": "Map signals suggest domain shift.",
        "sum": (
            "Δ-hotspots / low overlap indicate weak "
            "transfer support."
        ),
        "act": [
            "Treat transfer as exploratory (do not deploy).",
            "Inspect Δ-hotspots and buffered overlap.",
            "Prefer warm-start or baseline per city.",
        ],
        "opt": [
            "Enable Radar sweep hotspots.",
            "Enable Arrows between hotspots.",
            "Use Links mode=knn, k=2..3.",
        ],
    },
}


def interpret_transfer(
    run: Dict[str, Any],
    *,
    baseline_ref: Optional[Dict[str, Any]] = None,
    peer_runs: Optional[List[Dict[str, Any]]] = None,
    map_sig: Optional[Dict[str, Any]] = None,
    cfg: Optional[InterpretCfg] = None,
) -> Dict[str, Any]:
    """
    Interpret one Stage5 run dict.

    map_sig (optional) suggested fields:
    - dhot_n: int
    - dhot_abs_max: float
    - overlap_mean: float (0..1)
    - buf_n: int
    """
    cfg = cfg or InterpretCfg()

    strat = str(_get(run, "strategy", "xfer")).lower()
    split = str(_get(run, "split", "val")).lower()
    rm = str(_get(run, "rescale_mode", "as_is")).lower()
    cm = str(_get(run, "calibration", "none")).lower()

    mae = _f(_get(run, "overall_mae"))
    r2 = _f(_get(run, "overall_r2"))
    cov = _f(_get(run, "coverage80"))
    shp = _f(_get(run, "sharpness80"))

    schema = _get(run, "schema", {}) or {}
    st_al = _b(_get(schema, "static_aligned", False))
    d_om = _b(_get(schema, "dynamic_order_mismatch", False))
    f_om = _b(_get(schema, "future_order_mismatch", False))
    d_ro = _b(_get(schema, "dynamic_reordered", False))
    f_ro = _b(_get(schema, "future_reordered", False))

    sev = 0
    badges: List[str] = []
    acts: List[str] = []
    opts: List[str] = []
    crit: List[str] = []
    trig: List[Tuple[str, str]] = []

    def add(pid: str, why: str) -> None:
        nonlocal sev
        p = POLICIES[pid]
        sev = max(sev, int(p["sev"]))
        badges.append(str(p["badge"]))
        acts.extend(list(p.get("act", [])))
        opts.extend(list(p.get("opt", [])))
        trig.append((pid, why))

    # --- guards
    if mae is None and r2 is None and split == "test":
        add("missing_y_true", "test likely lacks y_true")

    # --- schema policies
    if st_al:
        add("static_shift", "static aligned across cities")
    if (d_om and not d_ro) or (f_om and not f_ro):
        add("order_mismatch", "order mismatch not fixed")

    # --- baseline sanity
    if strat == "baseline":
        if r2 is not None and r2 < cfg.r2_ok:
            add("weak_baseline", f"baseline r2={r2:.3g}")
        if mae is None and r2 is None and split != "test":
            add("weak_baseline", "baseline has no metrics")

    # --- general R2
    if r2 is not None and r2 < cfg.r2_ok:
        add("low_r2", f"r2={r2:.3g} < {cfg.r2_ok:g}")

    # --- coverage
    if cov is not None and cov < cfg.cov_low:
        add("undercoverage", f"cov80={cov:.3g}")
    if cov is not None and cov > cfg.cov_high and shp is not None:
        add("overwide", f"cov80={cov:.3g} high")

    # --- relative to baseline
    if baseline_ref is not None and strat in ("xfer", "warm"):
        b_mae = _f(_get(baseline_ref, "overall_mae"))
        b_shp = _f(_get(baseline_ref, "sharpness80"))

        if mae is not None and b_mae is not None and b_mae > 0:
            ratio = float(mae) / float(b_mae)
            if ratio >= cfg.mae_ratio_ok:
                add("transfer_drop", f"mae×{ratio:.2f}")

        if (
            b_shp is not None
            and shp is not None
            and b_shp > 0
            and cov is not None
            and cov >= cfg.cov_ok
        ):
            sratio = float(shp) / float(b_shp)
            if sratio >= cfg.sharp_ratio_bad:
                add("overwide", f"sharp×{sratio:.2f}")

        if strat == "warm" and peer_runs:
            best_xfer = None
            for pr in peer_runs:
                if str(_get(pr, "strategy", "")).lower() != "xfer":
                    continue
                pm = _f(_get(pr, "overall_mae"))
                if pm is None:
                    continue
                if best_xfer is None or pm < best_xfer:
                    best_xfer = pm
            if best_xfer is not None and mae is not None:
                if mae > best_xfer:
                    add("warm_not_helping", "warm worse than xfer")

    # --- map signals (Δ-hotspots / overlap / buffer)
    if map_sig:
        dh_n = _i(_get(map_sig, "dhot_n", 0)) or 0
        dh_m = _f(_get(map_sig, "dhot_abs_max"))
        ov_m = _f(_get(map_sig, "overlap_mean"))
        bf_n = _i(_get(map_sig, "buf_n", 0)) or 0

        shift = False
        if dh_n >= cfg.dhot_many:
            shift = True
        if dh_m is not None and dh_m >= cfg.dhot_abs_big:
            shift = True
        if ov_m is not None and ov_m <= cfg.overlap_low:
            shift = True
        if bf_n > 0 and bf_n <= cfg.buf_low_n:
            shift = True

        if shift:
            add("map_shift", "map_sig exceeds thresholds")

        # keep some critical notes (even if ok)
        if dh_n:
            crit.append(f"Δ-hotspots: n={dh_n}")
        if dh_m is not None:
            crit.append(f"Δ|max|={dh_m:.3g}")
        if ov_m is not None:
            crit.append(f"overlap≈{ov_m:.3g}")
        if bf_n:
            crit.append(f"buf_n={bf_n}")

    # --- craft headline / summary
    head = f"{strat.upper()} ({split})"
    summ = "No issues detected by policies."

    if trig:
        top = None
        for pid, _why in trig:
            if int(POLICIES[pid]["sev"]) == sev:
                top = pid
                break
        if top is None:
            top = trig[0][0]
        head = str(POLICIES[top]["head"])
        summ = str(POLICIES[top]["sum"])

    # --- numeric context
    nums: List[str] = []
    if mae is not None:
        nums.append(f"mae={mae:.4g}")
    if r2 is not None:
        nums.append(f"r2={r2:.4g}")
    if cov is not None:
        nums.append(f"cov80={cov:.3g}")
    if shp is not None:
        nums.append(f"sharp80={shp:.3g}")

    num_line = ", ".join(nums) or "metrics: NA"
    ctx = f"{head} · rm={rm}, calib={cm}"

    out = {
        "status": _status(int(sev)),
        "severity": int(sev),
        "badges": _uniq(badges),
        "headline": head,
        "summary": summ,
        "numbers": num_line,
        "context": ctx,
        "actions": _uniq(acts),
        "options": _uniq(opts),
        "critical": _uniq(crit),
        "triggers": [{"p": p, "why": w} for p, w in trig],
    }
    return out


def render_tip(p: Dict[str, Any]) -> str:
    """
    One-line guide under the Interpretation doc.
    """
    badges = p.get("badges", []) or []
    btxt = " · ".join(badges[:3])
    nums = p.get("numbers", "") or ""
    if btxt:
        return f"{nums}   [{btxt}]"
    return nums


def _status_badge(status: str) -> str:
    s = str(status or "").lower()
    if s == "fail":
        return "<span class='gpBadge gpFail'>FAIL</span>"
    if s == "warn":
        return "<span class='gpBadge gpWarn'>WARN</span>"
    if s == "info":
        return "<span class='gpBadge gpInfo'>INFO</span>"
    return "<span class='gpBadge gpOk'>OK</span>"


def _tags_html(tags: List[str]) -> str:
    out: List[str] = []
    for t in (tags or []):
        tt = str(t or "").strip()
        if tt:
            out.append(
                f"<span class='gpBadge gpTag'>{tt}</span>"
            )
    return " ".join(out)


def render_html(p: Dict[str, Any]) -> str:
    """
    Dynamic interpretation HTML.

    Note
    ----
    Do NOT include map_help_html() here because
    the Advanced panel already shows it by default.
    """
    badge = _status_badge(str(p.get("status", "")))
    tags = _tags_html(list(p.get("badges", []) or []))

    acts = list(p.get("actions", []) or [])
    opts = list(p.get("options", []) or [])
    crit = list(p.get("critical", []) or [])

    li_a = "".join(
        f"<li>{a}</li>" for a in acts if str(a).strip()
    )
    li_o = "".join(
        f"<li>{o}</li>" for o in opts if str(o).strip()
    )
    li_c = "".join(
        f"<li>{c}</li>" for c in crit if str(c).strip()
    )

    return (
        "<div style='line-height:1.25;'>"
        f"{badge}<b>{p.get('headline','')}</b><br>"
        f"<span>{p.get('summary','')}</span><br>"
        f"<span><i>{p.get('numbers','')}</i></span><br>"
        f"<span>{p.get('context','')}</span>"
        f"<div style='margin-top:6px;'>{tags}</div>"
        "<hr style='opacity:0.25;'>"
        "<b>Recommended actions</b>"
        f"<ul>{li_a or '<li>None</li>'}</ul>"
        "<b>Suggested map options</b>"
        f"<ul>{li_o or '<li>None</li>'}</ul>"
        "<b>Critical signals</b>"
        f"<ul>{li_c or '<li>None</li>'}</ul>"
        "</div>"
    )

def map_help_html() -> str:
    """
    Static guidance for the Interpretation panel.
    """
    return (
        "<div style='line-height:1.25;'>"
        "<b>How to read transfer maps</b>"
        "<ul>"
        "<li><b>Overlay</b>: A, B, or A+B points.</li>"
        "<li><b>Colors</b>: follow selected value; "
        "<b>Shared</b> compares magnitudes, "
        "<b>Auto</b> reveals within-city patterns.</li>"
        "<li><b>Hotspots</b>: extremes after quantile "
        "+ min-sep filtering.</li>"
        "</ul>"
        "<b>Shift signals (be cautious)</b>"
        "<ul>"
        "<li><b>Δ layer</b>: gridwise A−B differences "
        "(Interactions → Δ).</li>"
        "<li><b>Δ-hotspots</b>: dense/strong → likely "
        "domain shift.</li>"
        "<li><b>Overlap intensity</b>: low → weak shared "
        "support in space.</li>"
        "<li><b>Buffered intersection</b>: small → cities "
        "share little footprint.</li>"
        "</ul>"
        "<b>Tools to inspect</b>"
        "<ul>"
        "<li><b>Radar</b>: cycles through hotspot centers "
        "to review rank/score.</li>"
        "<li><b>Links</b>: arrows connect hotspot pairs "
        "(nearest/knn) to spot displacement.</li>"
        "</ul>"
        "<b>Recommended policy</b>"
        "<ul>"
        "<li>Start with <b>baseline</b> per city on val "
        "(must be reasonable).</li>"
        "<li>Use <b>xfer</b> only when shift signals are "
        "weak and overlap is good.</li>"
        "<li>Use <b>warm</b> to adapt on target when "
        "xfer drops vs baseline.</li>"
        "<li>Verify <b>Coords + EPSG</b> if points look "
        "displaced.</li>"
        "</ul>"
        "</div>"
    )
