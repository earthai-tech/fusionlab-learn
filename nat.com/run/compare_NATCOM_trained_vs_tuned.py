# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>
"""
compare_NATCOM_trained_vs_tuned.py

Compare key metrics between the *trained* and *tuned* GeoPriorSubsNet
runs for a given city/model.

It uses the same Stage-1 manifest discovery logic as the training and
tuning scripts, then:

- Finds the latest Stage-2 training run:  <run_dir>/train_YYYYMMDD-*
- Finds the latest Stage-2 tuning run:    <run_dir>/tuning/run_YYYYMMDD-*
- Loads:
    - geoprior_eval_phys_*.json                (trained)
    - geoprior_eval_phys_tuned_*.json          (tuned)
    - eval_diagnostics.json                    (trained)
    - eval_diagnostics_tuned.json              (tuned)
- Extracts:
    - Physical-space point metrics: MAE, MSE, R²
    - Interval metrics: coverage80, sharpness80
    - Per-horizon MAE/R²
    - Last-year & mean eval_diagnostics (MAE, MSE, R², PSS)
- Prints a side-by-side summary and writes a JSON summary into the
  tuned run directory:
    <tuned_run>/comparison_trained_vs_tuned.json

Usage
-----

    python compare_NATCOM_trained_vs_tuned.py \
        --city nansha \
        --model GeoPriorSubsNet

Optional arguments:

    --results-dir   Root results directory (defaults to default_results_dir())
    --stage1-manifest  Explicit path to Stage-1 manifest.json
    --train-run     Explicit Stage-2 training run directory
    --tune-run      Explicit Stage-2 tuning run directory

"""

from __future__ import annotations

import os
import json
import glob
import argparse
from typing import Any, Dict, Optional

import numpy as np

try:
    from fusionlab.api.util import get_table_size
    from fusionlab.utils.generic_utils import (
        default_results_dir,
        getenv_stripped,
        print_config_table,
    )
    from fusionlab.registry.utils import _find_stage1_manifest
except Exception as e:  # pragma: no cover - hard dependency for this script
    raise SystemExit(
        f"[Critical] Required fusionlab imports failed in "
        f"compare_NATCOM_trained_vs_tuned.py: {e}"
    )


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _load_json(path: str) -> Dict[str, Any]:
    """Load JSON gracefully."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _find_latest_subdir(parent: str, prefix: str) -> Optional[str]:
    """
    Find the most recent subdirectory under `parent` whose name starts
    with `prefix`. Returns absolute path or None if nothing is found.
    """
    if not os.path.isdir(parent):
        return None
    candidates = []
    for name in os.listdir(parent):
        if not name.startswith(prefix):
            continue
        full = os.path.join(parent, name)
        if os.path.isdir(full):
            candidates.append(full)
    if not candidates:
        return None
    # choose the most recently modified
    candidates.sort(key=os.path.getmtime)
    return candidates[-1]


def _pick_phys_json(run_dir: str, tuned: bool) -> Optional[str]:
    """
    Pick the geoprior_eval_phys JSON in a run directory.

    For trained runs:  geoprior_eval_phys_*.json
    For tuned runs:    geoprior_eval_phys_tuned_*.json
    """
    pattern = "geoprior_eval_phys_tuned_*.json" if tuned else "geoprior_eval_phys_*.json"
    paths = glob.glob(os.path.join(run_dir, pattern))
    if not paths:
        return None
    paths.sort(key=os.path.getmtime)
    return paths[-1]


def _extract_phys_summary(path: str) -> Dict[str, Any]:
    """
    Extract key fields from geoprior_eval_phys*.json.

    Expected structure (subset):

        {
          "point_metrics": {
             "mae": ...,
             "mse": ...,
             "r2":  ...
          },
          "coverage80": ...,
          "sharpness80": ...,
          "per_horizon": {
             "mae": {"H1": ..., "H2": ...},
             "r2":  {"H1": ..., "H2": ...}
          }
        }
    """
    if path is None:
        return {}

    j = _load_json(path)
    pt = j.get("point_metrics", {}) or {}

    per_h = j.get("per_horizon", {}) or {}
    per_h_mae = per_h.get("mae", {}) or {}
    per_h_r2 = per_h.get("r2", {}) or {}

    return {
        "mae": pt.get("mae"),
        "mse": pt.get("mse"),
        "r2": pt.get("r2"),
        "coverage80": j.get("coverage80"),
        "sharpness80": j.get("sharpness80"),
        "per_h_mae": per_h_mae,
        "per_h_r2": per_h_r2,
    }


def _extract_eval_diag_summary(path: str) -> Dict[str, Any]:
    """
    Extract last-year and mean eval_diagnostics summary.

    Expected structure (subset):

        {
          "2020.0": {
             "overall_mae": ...,
             "overall_mse": ...,
             "overall_r2": ...,
             "coverage80": ...,
             "sharpness80": ...,
             "pss": ...
          },
          "2021.0": { ... },
          ...
        }

    Returns
    -------
    {
      "last_year": 2022.0,
      "last": {
         "overall_mae": ...,
         "overall_mse": ...,
         "overall_r2": ...,
         "coverage80": ...,
         "sharpness80": ...,
         "pss": ...
      },
      "avg": {
         "overall_mae": ...,
         "overall_mse": ...,
         "overall_r2": ...,
         "coverage80": ...,
         "sharpness80": ...,
         "pss": ...
      }
    }
    """
    if path is None or not os.path.exists(path):
        return {}

    diag = _load_json(path)
    if not isinstance(diag, dict):
        return {}

    years = []
    for k in diag.keys():
        try:
            years.append(float(k))
        except (TypeError, ValueError):
            continue

    if not years:
        return {}

    years_sorted = sorted(years)
    last_year = years_sorted[-1]
    last = diag.get(str(last_year), {}) or {}

    keys = ["overall_mae", "overall_mse", "overall_r2", "coverage80", "sharpness80", "pss"]
    avg = {}
    for key in keys:
        vals = []
        for y in years_sorted:
            v = diag.get(str(y), {}).get(key)
            if v is not None:
                vals.append(float(v))
        avg[key] = float(np.mean(vals)) if vals else None

    return {
        "last_year": last_year,
        "last": {k: last.get(k) for k in keys},
        "avg": avg,
    }


def _delta(a: Optional[float], b: Optional[float]) -> Optional[float]:
    """
    Return b - a if both are not None, else None.
    """
    if a is None or b is None:
        return None
    try:
        return float(b) - float(a)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare trained vs tuned GeoPriorSubsNet results "
                    "for a given city/model."
    )
    parser.add_argument(
        "--results-dir",
        default=None,
        help="Root results directory (defaults to fusionlab default_results_dir()).",
    )
    parser.add_argument(
        "--city",
        default=None,
        help="City hint used when searching Stage-1 manifest "
             "(e.g., 'nansha', 'zhongshan').",
    )
    parser.add_argument(
        "--model",
        default="GeoPriorSubsNet",
        help="Model name hint used when searching Stage-1 manifest.",
    )
    parser.add_argument(
        "--stage1-manifest",
        default=None,
        help="Explicit path to Stage-1 manifest.json (optional).",
    )
    parser.add_argument(
        "--train-run",
        default=None,
        help="Explicit Stage-2 training run directory "
             "(e.g., .../train_20251119-210101).",
    )
    parser.add_argument(
        "--tune-run",
        default=None,
        help="Explicit Stage-2 tuning run directory "
             "(e.g., .../tuning/run_20251121-105529).",
    )

    args = parser.parse_args()

    # ---- Resolve Stage-1 manifest -----------------------------------
    results_dir = args.results_dir or default_results_dir()
    city_hint = args.city or getenv_stripped("CITY")
    model_hint = args.model or "GeoPriorSubsNet"
    manual_manifest = args.stage1_manifest

    manifest_path = _find_stage1_manifest(
        manual=manual_manifest,
        base_dir=results_dir,
        city_hint=city_hint,
        model_hint=model_hint,
        prefer="timestamp",
        required_keys=("model", "stage"),
        verbose=1,
    )
    M = _load_json(manifest_path)

    city_name = M.get("city", city_hint or "unknown_city")
    model_name = M.get("model", model_hint)

    base_run_dir = M["paths"]["run_dir"]

    # ---- Resolve Stage-2 training and tuning run dirs ---------------
    if args.train_run:
        train_run = os.path.abspath(args.train_run)
    else:
        train_run = _find_latest_subdir(base_run_dir, prefix="train_")

    tuning_root = os.path.join(base_run_dir, "tuning")
    if args.tune_run:
        tune_run = os.path.abspath(args.tune_run)
    else:
        tune_run = _find_latest_subdir(tuning_root, prefix="run_")

    if train_run is None:
        print(f"[Warn] Could not find any 'train_*' directory under: {base_run_dir}")
    if tune_run is None:
        print(f"[Warn] Could not find any 'run_*' directory under: {tuning_root}")

    if train_run is None or tune_run is None:
        raise SystemExit(
            "[Error] Missing trained or tuned run directory; "
            "cannot perform comparison."
        )

    print(f"\n[Info] Using training run: {train_run}")
    print(f"[Info] Using tuning   run: {tune_run}")

    # ---- Locate JSON artifacts --------------------------------------
    train_phys_json = _pick_phys_json(train_run, tuned=False)
    tune_phys_json = _pick_phys_json(tune_run, tuned=True)

    train_eval_diag_json = os.path.join(train_run, "eval_diagnostics.json")
    if not os.path.exists(train_eval_diag_json):
        train_eval_diag_json = None

    tune_eval_diag_json = os.path.join(tune_run, "eval_diagnostics_tuned.json")
    if not os.path.exists(tune_eval_diag_json):
        tune_eval_diag_json = None

    # ---- Extract summaries ------------------------------------------
    train_phys = _extract_phys_summary(train_phys_json)
    tune_phys = _extract_phys_summary(tune_phys_json)

    train_eval = _extract_eval_diag_summary(train_eval_diag_json)
    tune_eval = _extract_eval_diag_summary(tune_eval_diag_json)

    # ---- Build comparison payload -----------------------------------
    comparison: Dict[str, Any] = {
        "city": city_name,
        "model": model_name,
        "stage1_manifest": manifest_path,
        "train_run_dir": train_run,
        "tune_run_dir": tune_run,
        "trained": {
            "phys_json": train_phys_json,
            "eval_diag_json": train_eval_diag_json,
            "phys_summary": train_phys,
            "eval_diag_summary": train_eval,
        },
        "tuned": {
            "phys_json": tune_phys_json,
            "eval_diag_json": tune_eval_diag_json,
            "phys_summary": tune_phys,
            "eval_diag_summary": tune_eval,
        },
        "delta": {},
    }

    # Point metrics delta (tuned - trained)
    comparison["delta"]["point_metrics"] = {
        "mae": _delta(train_phys.get("mae"), tune_phys.get("mae")),
        "mse": _delta(train_phys.get("mse"), tune_phys.get("mse")),
        "r2": _delta(train_phys.get("r2"), tune_phys.get("r2")),
        "coverage80": _delta(train_phys.get("coverage80"), tune_phys.get("coverage80")),
        "sharpness80": _delta(train_phys.get("sharpness80"), tune_phys.get("sharpness80")),
    }

    # Eval-diagnostics last-year averages delta
    if train_eval and tune_eval:
        t_last = train_eval.get("last", {})
        u_last = tune_eval.get("last", {})
        t_avg = train_eval.get("avg", {})
        u_avg = tune_eval.get("avg", {})

        last_delta = {
            k: _delta(t_last.get(k), u_last.get(k))
            for k in ("overall_mae", "overall_mse", "overall_r2",
                      "coverage80", "sharpness80", "pss")
        }
        avg_delta = {
            k: _delta(t_avg.get(k), u_avg.get(k))
            for k in ("overall_mae", "overall_mse", "overall_r2",
                      "coverage80", "sharpness80", "pss")
        }
        comparison["delta"]["eval_last"] = last_delta
        comparison["delta"]["eval_avg"] = avg_delta

    # ---- Pretty-print via print_config_table ------------------------
    sections = []

    # 1. Point metrics (physical)
    sections.append((
        "Point metrics (physical, geoprior_eval_phys*.json)",
        {
            "TRAIN_mae": train_phys.get("mae"),
            "TUNED_mae": tune_phys.get("mae"),
            "Δ_mae (tuned - trained)": comparison["delta"]["point_metrics"]["mae"],
            "TRAIN_mse": train_phys.get("mse"),
            "TUNED_mse": tune_phys.get("mse"),
            "Δ_mse": comparison["delta"]["point_metrics"]["mse"],
            "TRAIN_r2": train_phys.get("r2"),
            "TUNED_r2": tune_phys.get("r2"),
            "Δ_r2": comparison["delta"]["point_metrics"]["r2"],
        }
    ))

    # 2. Interval metrics
    sections.append((
        "Interval metrics (coverage80 / sharpness80)",
        {
            "TRAIN_coverage80": train_phys.get("coverage80"),
            "TUNED_coverage80": tune_phys.get("coverage80"),
            "Δ_coverage80": comparison["delta"]["point_metrics"]["coverage80"],
            "TRAIN_sharpness80": train_phys.get("sharpness80"),
            "TUNED_sharpness80": tune_phys.get("sharpness80"),
            "Δ_sharpness80": comparison["delta"]["point_metrics"]["sharpness80"],
        }
    ))

    # 3. Eval diagnostics (last year)
    if train_eval and tune_eval:
        ty = train_eval["last_year"]
        uy = tune_eval["last_year"]
        t_last = train_eval["last"]
        u_last = tune_eval["last"]
        sections.append((
            f"Eval diagnostics (last year: train {ty}, tuned {uy})",
            {
                "TRAIN_last_overall_mae": t_last.get("overall_mae"),
                "TUNED_last_overall_mae": u_last.get("overall_mae"),
                "Δ_last_overall_mae": comparison["delta"]["eval_last"]["overall_mae"],
                "TRAIN_last_overall_r2": t_last.get("overall_r2"),
                "TUNED_last_overall_r2": u_last.get("overall_r2"),
                "Δ_last_overall_r2": comparison["delta"]["eval_last"]["overall_r2"],
                "TRAIN_last_pss": t_last.get("pss"),
                "TUNED_last_pss": u_last.get("pss"),
                "Δ_last_pss": comparison["delta"]["eval_last"]["pss"],
            }
        ))

        # 4. Eval diagnostics (mean across years)
        t_avg = train_eval["avg"]
        u_avg = tune_eval["avg"]
        sections.append((
            "Eval diagnostics (mean across years)",
            {
                "TRAIN_avg_overall_mae": t_avg.get("overall_mae"),
                "TUNED_avg_overall_mae": u_avg.get("overall_mae"),
                "Δ_avg_overall_mae": comparison["delta"]["eval_avg"]["overall_mae"],
                "TRAIN_avg_overall_r2": t_avg.get("overall_r2"),
                "TUNED_avg_overall_r2": u_avg.get("overall_r2"),
                "Δ_avg_overall_r2": comparison["delta"]["eval_avg"]["overall_r2"],
                "TRAIN_avg_pss": t_avg.get("pss"),
                "TUNED_avg_pss": u_avg.get("pss"),
                "Δ_avg_pss": comparison["delta"]["eval_avg"]["pss"],
            }
        ))

    title = f"{city_name.upper()} {model_name} — TRAINED vs TUNED"
    print()
    print_config_table(
        sections,
        table_width=get_table_size(),
        title=title,
    )

    # ---- Save JSON summary in tuned run directory -------------------
    out_json = os.path.join(tune_run, "comparison_trained_vs_tuned.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)
    print(f"\n[OK] Comparison summary written to:\n    {out_json}\n")


if __name__ == "__main__":
    main()
