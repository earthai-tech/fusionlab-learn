# geoprior/ui/map/model_eval_interpretation.py
# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


_EVAL_PATS = (
    "geoprior_eval*_interpretable.json",
    "geoprior_eval*.json",
)


@dataclass(frozen=True)
class ModelBlock:
    id: str
    title: str
    level: str  # "ok" | "watch" | "warn"
    implication: str
    action: str
    sdgs: Tuple[str, ...] = ()
    evidence: Tuple[str, ...] = ()


def find_eval_json_near(
    data_path: Path,
) -> Optional[Path]:
    p = Path(data_path)
    for d in (p.parent, p.parent.parent):
        if not d or not d.exists():
            continue
        for pat in _EVAL_PATS:
            hits = sorted(d.glob(pat))
            if hits:
                return hits[0]
    return None


def load_eval_json(path: Path) -> Dict[str, Any]:
    with open(str(path), "r", encoding="utf-8") as f:
        return dict(json.load(f) or {})


def build_model_blocks(
    obj: Dict[str, Any],
) -> List[ModelBlock]:
    units = dict(obj.get("units") or {})
    mets = dict(obj.get("metrics_evaluate") or {})
    diag = dict(obj.get("physics_diagnostics") or {})
    cal = dict(obj.get("interval_calibration") or {})
    censor = dict(obj.get("censor_stratified") or {})
    pm = dict(obj.get("point_metrics") or {})

    subs_u = str(units.get("subs_metrics_unit") or "").strip()

    blocks: List[ModelBlock] = []

    # ---- Accuracy block
    mae = _f(mets.get("subs_pred_mae_q50", pm.get("mae")))
    rmse = _f(mets.get("subs_pred_rmse_q50", pm.get("rmse")))
    r2 = _f(pm.get("r2", None))
    if mae is not None or rmse is not None:
        ev = []
        if mae is not None:
            ev.append(_kv("MAE", mae, subs_u))
        if rmse is not None:
            ev.append(_kv("RMSE", rmse, subs_u))
        if r2 is not None:
            ev.append(_kv("R²", r2, ""))

        blocks.append(
            ModelBlock(
                id="accuracy",
                title="Forecast accuracy (subsidence)",
                level="ok",
                implication=(
                    "Overall fit quality for subsidence forecasts."
                ),
                action=(
                    "Use higher caution where local residuals are "
                    "large; prioritize validation in critical assets."
                ),
                sdgs=("SDG 11", "SDG 9"),
                evidence=tuple(ev),
            )
        )

    # ---- Interval calibration block
    tgt = _f(cal.get("target", 0.8))
    cov_c = _f(cal.get("coverage80_calibrated", None))
    shp_c = _f(cal.get("sharpness80_calibrated", None))
    if cov_c is not None:
        lvl = _level_cov(cov_c, tgt)
        ev = [
            _kv("Target", tgt, ""),
            _kv("Coverage80 (cal)", cov_c, ""),
        ]
        if shp_c is not None:
            ev.append(_kv("Sharpness80 (cal)", shp_c, ""))

        blocks.append(
            ModelBlock(
                id="intervals",
                title="Prediction interval calibration",
                level=lvl,
                implication=(
                    "How reliable the uncertainty bands are "
                    "for planning and safety margins."
                ),
                action=(
                    "Allocate monitoring to high-uncertainty "
                    "zones; design safety margins using the "
                    "interval width."
                ),
                sdgs=("SDG 11", "SDG 13"),
                evidence=tuple(ev),
            )
        )

    # ---- Physics diagnostics block(s)
    ep = _f(diag.get("epsilon_prior", mets.get("epsilon_prior")))
    ec = _f(diag.get("epsilon_cons", mets.get("epsilon_cons")))
    eg = _f(diag.get("epsilon_gw", mets.get("epsilon_gw")))

    if ep is not None:
        blocks.append(
            ModelBlock(
                id="eps_prior",
                title="Physics adequacy (ε_prior)",
                level=_level_eps(ep, good=0.01, warn=0.03),
                implication=(
                    "Where consolidation prior assumptions may "
                    "be inadequate or biased."
                ),
                action=(
                    "Commission detailed hydro-geomechanical "
                    "studies or higher-resolution models in "
                    "high-ε-prior areas."
                ),
                sdgs=("SDG 6", "SDG 13"),
                evidence=(f"ε_prior={ep:.6g}",),
            )
        )

    if ec is not None:
        u = str(units.get("epsilon_cons_raw_unit") or "").strip()
        ecr = _f(mets.get("epsilon_cons_raw", None))
        ev = [f"ε_cons={ec:.6g}"]
        if ecr is not None:
            ev.append(_kv("ε_cons_raw", ecr, u))

        blocks.append(
            ModelBlock(
                id="eps_cons",
                title="Dynamics fit (ε_cons)",
                level=_level_eps(ec, good=1e-4, warn=1e-3),
                implication=(
                    "Potential dynamics misspecification or data "
                    "quality issues in some locations/horizons."
                ),
                action=(
                    "Enhance InSAR/well monitoring; revisit "
                    "drivers and model structure in high-ε_cons "
                    "zones; treat forecasts as lower confidence."
                ),
                sdgs=("SDG 11", "SDG 6"),
                evidence=tuple(ev),
            )
        )

    if eg is not None:
        blocks.append(
            ModelBlock(
                id="eps_gw",
                title="Groundwater residual (ε_gw)",
                level=_level_eps(eg, good=1e-5, warn=1e-4),
                implication=(
                    "How consistent groundwater dynamics are with "
                    "the learned physical closures."
                ),
                action=(
                    "If ε_gw is elevated, strengthen groundwater "
                    "observations and reconsider boundary "
                    "conditions and aquifer parameter priors."
                ),
                sdgs=("SDG 6", "SDG 13"),
                evidence=(f"ε_gw={eg:.6g}",),
            )
        )

    # ---- Bias / censoring block
    if censor:
        unc = _f(censor.get("mae_uncensored", None))
        cen = _f(censor.get("mae_censored", None))
        if unc is not None or cen is not None:
            ev = []
            if cen is not None:
                ev.append(_kv("MAE (censored)", cen, subs_u))
            if unc is not None:
                ev.append(_kv("MAE (uncensored)", unc, subs_u))

            blocks.append(
                ModelBlock(
                    id="bias",
                    title="Censor / data-coverage sensitivity",
                    level="watch",
                    implication=(
                        "Performance may differ between well-"
                        "observed and poorly observed zones."
                    ),
                    action=(
                        "Target data acquisition in under-"
                        "instrumented areas to reduce bias."
                    ),
                    sdgs=("SDG 11", "SDG 9"),
                    evidence=tuple(ev),
                )
            )

    return blocks


def render_blocks_md(
    blocks: Sequence[ModelBlock],
    *,
    selected: Iterable[str],
    heading: str = "## Model-driven interpretation",
) -> str:
    sel = set(str(s) for s in (selected or []))
    keep = [b for b in blocks if b.id in sel]
    if not keep:
        return ""

    lines: List[str] = [heading, ""]
    for b in keep:
        lines.append(f"### {b.title}")
        lines.append(f"- Level: **{b.level}**")
        if b.evidence:
            lines.append("- Evidence: " + "; ".join(b.evidence))
        if b.sdgs:
            lines.append("- SDGs: " + ", ".join(b.sdgs))
        lines.append(f"- Implication: {b.implication}")
        lines.append(f"- Action: {b.action}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


# ----------------- small helpers -----------------

def _f(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        return None


def _kv(k: str, v: float, u: str) -> str:
    uu = str(u or "").strip()
    if uu:
        return f"{k}={v:.6g} {uu}"
    return f"{k}={v:.6g}"


def _level_cov(cov: float, tgt: float) -> str:
    d = abs(float(cov) - float(tgt))
    if d <= 0.03:
        return "ok"
    if d <= 0.08:
        return "watch"
    return "warn"


def _level_eps(e: float, *, good: float, warn: float) -> str:
    x = float(e)
    if x <= good:
        return "ok"
    if x <= warn:
        return "watch"
    return "warn"
