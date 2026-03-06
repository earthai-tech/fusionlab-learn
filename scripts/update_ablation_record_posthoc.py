#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Update ablation_record.jsonl using post-hoc calibrated metrics.

What this does
--------------
For each JSONL row (one run):
- Finds the run folder via the timestamp:
  geoprior_eval_phys_<TS>_interpretable.json
- Replaces top-level metrics with *post-hoc calibrated* values:
  mae, mse, rmse, r2, coverage80, sharpness80, per-horizon metrics
- Writes units + detailed blocks into record["metrics"]
- Stores the old (pre-patch) values under:
  record["legacy"]["pre_posthoc_patch"]

Typical layout
--------------
results_root/
  run_folder_A/
    geoprior_eval_phys_<TS>_interpretable.json
    <city>_GeoPriorSubsNet_eval_diagnostics_TestSet_H3_calibrated.json
    ablations_records/ablation_record.jsonl
  run_folder_B/
    ...

Note that ablations_records/ folder is under each run_folder. 
ablation_records/
  ablation_record.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


_INTERP_PREFIX = "geoprior_eval_phys_"
_INTERP_SUFFIX = "_interpretable.json"


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        return None


def _pick_first(*vals: Any) -> Any:
    for v in vals:
        if v is not None:
            return v
    return None


def _build_ts_index(runs_root: Path) -> Dict[str, Path]:
    """
    Map timestamp -> run_dir by scanning interpretable phys JSONs.
    """
    idx: Dict[str, Path] = {}
    pat = f"{_INTERP_PREFIX}*{_INTERP_SUFFIX}"
    for p in runs_root.rglob(pat):
        name = p.name
        if not (name.startswith(_INTERP_PREFIX) and
                name.endswith(_INTERP_SUFFIX)):
            continue
        ts = name[len(_INTERP_PREFIX) : -len(_INTERP_SUFFIX)]
        idx[ts] = p.parent
    return idx


def _find_diag_calibrated(
    run_dir: Path,
    city: Optional[str],
) -> Optional[Path]:
    """
    Prefer <city>_*eval_diagnostics*_calibrated.json.
    Fallback to any *eval_diagnostics*_calibrated.json.
    """
    if city:
        cands = list(
            run_dir.rglob(
                f"{city}_*eval_diagnostics*_calibrated.json"
            )
        )
        if cands:
            return cands[0]

    cands = list(run_dir.rglob("*eval_diagnostics*_calibrated.json"))
    if cands:
        return cands[0]
    return None


def _hkeys_to_Hn(d: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert {"1": v, "2": v} -> {"H1": v, "H2": v}.
    Leaves keys that already look like "H1" unchanged.
    """
    out: Dict[str, Any] = {}
    for k, v in (d or {}).items():
        ks = str(k)
        if ks.startswith("H"):
            out[ks] = v
        else:
            out[f"H{ks}"] = v
    return out


def _update_one_record(
    rec: Dict[str, Any],
    run_dir: Path,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    ts = str(rec.get("timestamp") or "")
    city = rec.get("city")

    interp_path = run_dir / f"{_INTERP_PREFIX}{ts}{_INTERP_SUFFIX}"
    if not interp_path.exists():
        return rec, {
            "ok": False,
            "reason": "missing_interpretable_json",
            "expected": str(interp_path),
        }

    interp = _read_json(interp_path)

    diag_cal_path = _find_diag_calibrated(run_dir, city)
    diag_cal = _read_json(diag_cal_path) if diag_cal_path else {}
    overall = (diag_cal or {}).get("__overall__", {})

    # --- Preserve old (wrong) headline metrics ---
    legacy: Dict[str, Any] = {}
    for k in ["r2", "mse", "mae", "rmse", "coverage80", "sharpness80"]:
        if k in rec:
            legacy[k] = rec.get(k)
    rec.setdefault("legacy", {})
    rec["legacy"]["pre_posthoc_patch"] = legacy

    # --- Units (interpretable JSON is authoritative) ---
    units = interp.get("units", {}) or {}
    rec["units"] = units

    # --- Post-hoc point metrics (prefer diagnostics overall, then interp) ---
    p_point = interp.get("point_metrics", {}) or {}
    r2 = _pick_first(overall.get("overall_r2"), p_point.get("r2"))
    mse = _pick_first(overall.get("overall_mse"), p_point.get("mse"))
    mae = _pick_first(overall.get("overall_mae"), p_point.get("mae"))
    rmse = _pick_first(overall.get("overall_rmse"), p_point.get("rmse"))
    if rmse is None and mse is not None:
        try:
            rmse = math.sqrt(float(mse))
        except Exception:
            rmse = None

    # --- Post-hoc calibrated interval metrics ---
    ival = interp.get("interval_calibration", {}) or {}
    cov_cal = _pick_first(
        overall.get("coverage80"),
        ival.get("coverage80_calibrated_phys"),
        ival.get("coverage80_calibrated"),
    )
    shp_cal = _pick_first(
        overall.get("sharpness80"),
        ival.get("sharpness80_calibrated_phys"),
        ival.get("sharpness80_calibrated"),
    )

    # --- Uncalibrated (for transparency/debug) ---
    cov_uncal = _pick_first(
        ival.get("coverage80_uncalibrated_phys"),
        ival.get("coverage80_uncalibrated"),
    )
    shp_uncal = _pick_first(
        ival.get("sharpness80_uncalibrated_phys"),
        ival.get("sharpness80_uncalibrated"),
    )

    # --- Per-horizon (prefer diagnostics overall, then interp) ---
    ph_mae = overall.get("per_horizon_mae") or {}
    ph_r2 = overall.get("per_horizon_r2") or {}
    if ph_mae:
        rec["per_horizon_mae"] = _hkeys_to_Hn(ph_mae)
    else:
        rec["per_horizon_mae"] = (
            (interp.get("per_horizon", {}) or {}).get("mae")
        )

    if ph_r2:
        rec["per_horizon_r2"] = _hkeys_to_Hn(ph_r2)
    else:
        rec["per_horizon_r2"] = (
            (interp.get("per_horizon", {}) or {}).get("r2")
        )

    # --- Physics diagnostics (interpretable JSON) ---
    phys = interp.get("physics_diagnostics", {}) or {}
    for k in ["epsilon_prior", "epsilon_cons", "epsilon_gw"]:
        if k in phys:
            rec[k] = phys.get(k)

    # --- Write updated headline metrics (paper-consistent) ---
    rec["r2"] = _safe_float(r2)
    rec["mse"] = _safe_float(mse)
    rec["mae"] = _safe_float(mae)
    rec["rmse"] = _safe_float(rmse)
    rec["coverage80"] = _safe_float(cov_cal)
    rec["sharpness80"] = _safe_float(shp_cal)

    # --- Attach full blocks for traceability ---
    rec["metrics"] = {
        "posthoc": {
            "point_metrics": {
                "mae": _safe_float(mae),
                "mse": _safe_float(mse),
                "rmse": _safe_float(rmse),
                "r2": _safe_float(r2),
            },
            "interval": {
                "target": ival.get("target"),
                "coverage80_calibrated": _safe_float(cov_cal),
                "sharpness80_calibrated": _safe_float(shp_cal),
                "coverage80_uncalibrated": _safe_float(cov_uncal),
                "sharpness80_uncalibrated": _safe_float(shp_uncal),
                "factors_per_horizon": ival.get("factors_per_horizon"),
            },
            "per_horizon": interp.get("per_horizon", {}),
        },
        "raw": {
            "metrics_evaluate": interp.get("metrics_evaluate", {}),
        },
        "sources": {
            "interpretable": str(interp_path),
            "eval_diag_cal": (
                str(diag_cal_path) if diag_cal_path else None
            ),
        },
    }

    return rec, {
        "ok": True,
        "ts": ts,
        "run_dir": str(run_dir),
        "unit": units.get("subs_metrics_unit"),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--runs-root",
        type=Path,
        required=True,
        help="Folder containing run folders (scan recursive).",
    )
    ap.add_argument(
        "--ablation",
        type=Path,
        required=True,
        help="Path to ablation_record.jsonl",
    )
    ap.add_argument(
        "--backup",
        action="store_true",
        help="Write a .bak copy before overwriting.",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not write; just report.",
    )
    args = ap.parse_args()

    runs_root = args.runs_root.expanduser().resolve()
    ab_path = args.ablation.expanduser().resolve()

    if not ab_path.exists():
        raise SystemExit(f"Missing: {ab_path}")

    idx = _build_ts_index(runs_root)
    if not idx:
        raise SystemExit(
            "No geoprior_eval_phys_*_interpretable.json found."
        )

    lines = ab_path.read_text(encoding="utf-8").splitlines()
    recs: list[Dict[str, Any]] = []
    infos: list[Dict[str, Any]] = []

    for ln in lines:
        if not ln.strip():
            continue
        rec = json.loads(ln)
        ts = str(rec.get("timestamp") or "")
        run_dir = idx.get(ts)
        if run_dir is None:
            infos.append(
                {"ok": False, "ts": ts, "reason": "ts_not_found"}
            )
            recs.append(rec)
            continue

        rec_u, info = _update_one_record(rec, run_dir)
        recs.append(rec_u)
        infos.append(info)

    ok_n = sum(1 for i in infos if i.get("ok"))
    bad_n = len(infos) - ok_n

    print(f"Updated: {ok_n} | Skipped: {bad_n}")
    for i in infos:
        if not i.get("ok"):
            print(f" - {i}")

    if args.dry_run:
        return 0

    if args.backup:
        bak = ab_path.with_suffix(ab_path.suffix + ".bak")
        bak.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"Backup: {bak}")

    out_lines = [
        json.dumps(r, ensure_ascii=False)
        for r in recs
    ]
    ab_path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    print(f"Wrote: {ab_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())