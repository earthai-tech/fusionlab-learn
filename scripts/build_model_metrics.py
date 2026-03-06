# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio
#
# scripts/build_model_metrics.py
#
"""Build a unified "model metrics" table from GeoPrior runs.

This scans a results root (or a single run folder) and collects
metrics from ablation records, preferring *updated* JSONL records.

Outputs (under scripts/out/ unless overridden)
--------------------------------------------
- <out>.csv        (wide table; one row per run)
- <out>.json       (same as CSV, as a list of dicts)
- <out>_long.csv   (optional; one row per horizon per run)
- <out>_long.json  (optional)

Data sources
------------
We scan for JSONL under:
  <src>/**/ablation_records/ablation_record*.jsonl

Preference order (robust even if cfg differs)
--------------------------------------------
  1) ablation_record.updated*.jsonl
  2) ablation_record*.jsonl

Notes
-----
- Legacy (SI) ablation records are auto-converted to mm when
  values look like meters (heuristic).
- Per-horizon blocks are expanded to columns like:
    r2_H1, r2_H2, mae_H1, mae_H2, ...
- Interval calibration fields are exported when present:
    coverage80_cal_phys, sharpness80_cal_phys, ...
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from . import config as cfg
from . import utils


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def _parse_args(argv: Optional[List[str]]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="build-model-metrics",
        description="Build a unified model metrics table.",
    )

    p.add_argument(
        "--src",
        "--results-root",
        "--run-dir",
        dest="src",
        type=str,
        default="results",
        help=(
            "Results root (scan recursive) OR a single run dir."
        ),
    )
    p.add_argument(
        "--out",
        type=str,
        default="model_metrics",
        help=(
            "Output stem/path (scripts/out/ if relative). "
            "Suffix is ignored."
        ),
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Optional output dir override.",
    )

    utils.add_city_flags(p, default_both=False)
    p.add_argument(
        "--city",
        type=str,
        default="",
        help=(
            "Extra city filter (comma list). "
            "Example: --city Nansha,Zhongshan"
        ),
    )

    p.add_argument(
        "--models",
        type=str,
        default="",
        help=(
            "Comma list of model names to keep (exact match)."
        ),
    )

    p.add_argument(
        "--include-long",
        type=str,
        default="true",
        help="Write a per-horizon long table (true/false).",
    )

    p.add_argument(
        "--dedupe",
        type=str,
        default="true",
        help=(
            "Deduplicate by (timestamp, city, model) "
            "preferring updated (true/false)."
        ),
    )

    return p.parse_args(argv)


# ---------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------

def _read_jsonl(fp: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with fp.open("r", encoding="utf-8") as f:
        for ln in f:
            s = ln.strip()
            if not s:
                continue
            try:
                rec = json.loads(s)
            except Exception:
                continue
            if isinstance(rec, dict):
                rows.append(rec)
    return rows


def _preferred_ablation_patterns() -> Tuple[str, ...]:
    """Enforce updated-first precedence, even if cfg differs."""
    base = list(cfg.PATTERNS.get("ablation_record_jsonl", ()))
    prefer = [
        "ablation_records/ablation_record.updated*.jsonl",
        "ablation_records/ablation_record*.jsonl",
        "ablation_record.updated*.jsonl",
        "ablation_record*.jsonl",
    ]
    out: List[str] = []
    for pat in prefer + base:
        if pat and pat not in out:
            out.append(pat)
    return tuple(out)


def _scan_ablation_jsonl(src: Path) -> List[Dict[str, Any]]:
    pats = _preferred_ablation_patterns()
    files = utils.find_all(src, pats, must_exist=False)
    rows: List[Dict[str, Any]] = []

    for fp in files:
        try:
            recs = _read_jsonl(fp)
        except Exception:
            continue

        # infer run_dir from path:
        # <run_dir>/ablation_records/ablation_record*.jsonl
        run_dir = fp.parent.parent
        for r in recs:
            rr = dict(r)
            rr["_src"] = str(fp)
            rr.setdefault("_run_dir", str(run_dir))
            rows.append(rr)

    return rows


# ---------------------------------------------------------------------
# Canon / scaling
# ---------------------------------------------------------------------

def _hkeys_to_Hn(d: Any) -> Dict[str, Any]:
    if not isinstance(d, dict):
        return {}
    out: Dict[str, Any] = {}
    for k, v in d.items():
        ks = str(k)
        if ks.startswith("H"):
            out[ks] = v
        else:
            out[f"H{ks}"] = v
    return out


def _get_nested(d: Any, path: Tuple[str, ...]) -> Any:
    cur: Any = d
    for k in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(k, None)
    return cur


def _to_float(x: Any) -> float:
    try:
        v = float(x)
    except Exception:
        return float("nan")
    return v


def _needs_si_to_mm(rec: Dict[str, Any]) -> bool:
    """Heuristic for legacy SI ablation records."""
    u = rec.get("units", None)
    if isinstance(u, dict):
        uu = str(u.get("subs_metrics_unit", "")).lower()
        if uu == "mm":
            return False
        if uu in {"m", "meter", "metre"}:
            return True

    mae = _to_float(rec.get("mae", np.nan))
    mse = _to_float(rec.get("mse", np.nan))
    shp = _to_float(rec.get("sharpness80", np.nan))

    ok = (
        np.isfinite(mae)
        and np.isfinite(mse)
        and np.isfinite(shp)
        and 0.0 < mae < 1.0
        and 0.0 < mse < 1.0
        and 0.0 < shp < 1.0
    )
    return bool(ok)


def _scale_legacy_to_mm(rec: Dict[str, Any]) -> Dict[str, Any]:
    if not _needs_si_to_mm(rec):
        return rec

    out = dict(rec)

    for k in ["mae", "rmse", "sharpness80"]:
        v = out.get(k, None)
        if isinstance(v, (int, float)) and np.isfinite(v):
            out[k] = float(v) * 1000.0

    v = out.get("mse", None)
    if isinstance(v, (int, float)) and np.isfinite(v):
        out["mse"] = float(v) * 1e6

    u = dict((out.get("units") or {}))
    u.setdefault("subs_metrics_unit", "mm")
    u.setdefault("subs_factor_si_to_real", 1000.0)
    out["units"] = u
    return out


def _record_score(rec: Dict[str, Any]) -> int:
    s = 0
    src = str(rec.get("_src", "")).lower()
    if "updated" in src:
        s += 100
    u = rec.get("units", None)
    if isinstance(u, dict):
        uu = str(u.get("subs_metrics_unit", "")).lower()
        if uu == "mm":
            s += 50
    if isinstance(rec.get("metrics", None), dict):
        s += 10
    for k in ["rmse", "pss", "epsilon_prior", "epsilon_cons"]:
        v = _to_float(rec.get(k, np.nan))
        if np.isfinite(v):
            s += 1
    return int(s)


def _dedupe_records(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not rows:
        return rows

    key_map: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    for r0 in rows:
        r = _scale_legacy_to_mm(r0)
        ts = str(r.get("timestamp") or "")
        city = utils.canonical_city(str(r.get("city") or ""))
        model = str(r.get("model") or "")
        key = (ts, city, model)

        best = key_map.get(key, None)
        if best is None:
            r["city"] = city
            key_map[key] = r
            continue

        if _record_score(r) > _record_score(best):
            r["city"] = city
            key_map[key] = r

    return list(key_map.values())


# ---------------------------------------------------------------------
# Flatten to table
# ---------------------------------------------------------------------

def _extract_interval_cols(rec: Dict[str, Any]) -> Dict[str, Any]:
    """Pull calibrated/uncalibrated interval metrics when present."""
    out: Dict[str, Any] = {}

    ival = _get_nested(
        rec,
        ("metrics", "posthoc", "interval_calibration"),
    )
    if not isinstance(ival, dict):
        ival = _get_nested(rec, ("metrics", "posthoc", "interval"))
    if not isinstance(ival, dict):
        ival = rec.get("interval_calibration", None)

    if not isinstance(ival, dict):
        return out

    def _pick(*ks: str) -> Any:
        for k in ks:
            if k in ival:
                return ival.get(k)
        return None

    out["interval_target"] = _pick("target")

    # scaled
    out["coverage80_uncal"] = _pick("coverage80_uncalibrated")
    out["coverage80_cal"] = _pick("coverage80_calibrated")
    out["sharpness80_uncal"] = _pick("sharpness80_uncalibrated")
    out["sharpness80_cal"] = _pick("sharpness80_calibrated")

    # physical
    out["coverage80_uncal_phys"] = _pick(
        "coverage80_uncalibrated_phys"
    )
    out["coverage80_cal_phys"] = _pick(
        "coverage80_calibrated_phys"
    )
    out["sharpness80_uncal_phys"] = _pick(
        "sharpness80_uncalibrated_phys"
    )
    out["sharpness80_cal_phys"] = _pick(
        "sharpness80_calibrated_phys"
    )

    fac = _pick("factors_per_horizon")
    if fac is not None:
        try:
            out["factors_per_horizon"] = json.dumps(fac)
        except Exception:
            out["factors_per_horizon"] = str(fac)

    stats = _pick("factors_per_horizon_from_cal_stats")
    if isinstance(stats, dict):
        eb = stats.get("eval_before", {}) or {}
        ea = stats.get("eval_after", {}) or {}
        phb = eb.get("per_horizon", {}) or {}
        pha = ea.get("per_horizon", {}) or {}

        for hk, hv in (phb or {}).items():
            if not isinstance(hv, dict):
                continue
            H = f"H{hk}" if not str(hk).startswith("H") else str(hk)
            out[f"coverage80_uncal_{H}"] = hv.get("coverage")
            out[f"sharpness80_uncal_{H}"] = hv.get("sharpness")

        for hk, hv in (pha or {}).items():
            if not isinstance(hv, dict):
                continue
            H = f"H{hk}" if not str(hk).startswith("H") else str(hk)
            out[f"coverage80_cal_{H}"] = hv.get("coverage")
            out[f"sharpness80_cal_{H}"] = hv.get("sharpness")

    return out


def _extract_per_horizon_cols(rec: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}

    pr2 = rec.get("per_horizon_r2", None)
    pmae = rec.get("per_horizon_mae", None)

    if not isinstance(pr2, dict):
        pr2 = _get_nested(rec, ("per_horizon", "r2"))
    if not isinstance(pmae, dict):
        pmae = _get_nested(rec, ("per_horizon", "mae"))

    pr2 = _hkeys_to_Hn(pr2)
    pmae = _hkeys_to_Hn(pmae)

    for hk, hv in pr2.items():
        out[f"r2_{hk}"] = hv
    for hk, hv in pmae.items():
        out[f"mae_{hk}"] = hv

    return out


def _record_to_row(rec0: Dict[str, Any]) -> Dict[str, Any]:
    rec = _scale_legacy_to_mm(rec0)
    out: Dict[str, Any] = {}

    # id + config
    for k in [
        "timestamp",
        "city",
        "model",
        "pde_mode",
        "use_effective_h",
        "kappa_mode",
        "hd_factor",
        "lambda_cons",
        "lambda_gw",
        "lambda_prior",
        "lambda_smooth",
        "lambda_mv",
    ]:
        if k in rec:
            out[k] = rec.get(k)

    out["city"] = utils.canonical_city(str(out.get("city") or ""))

    # headline metrics
    for k in [
        "r2",
        "mae",
        "mse",
        "rmse",
        "pss",
        "coverage80",
        "sharpness80",
        "epsilon_prior",
        "epsilon_cons",
        "epsilon_gw",
        "epsilon_cons_raw",
        "epsilon_gw_raw",
    ]:
        if k in rec:
            out[k] = rec.get(k)

    # rmse fallback
    if (
        ("rmse" not in out or out.get("rmse") is None)
        and ("mse" in out)
    ):
        try:
            out["rmse"] = float(out["mse"]) ** 0.5
        except Exception:
            pass

    # units (if present)
    u = rec.get("units", None)
    if isinstance(u, dict):
        out["subs_metrics_unit"] = u.get("subs_metrics_unit")
        out["time_units"] = u.get("time_units")

    # provenance
    out["_src"] = rec.get("_src")
    out["_run_dir"] = rec.get("_run_dir")

    # expanded blocks
    out.update(_extract_interval_cols(rec))
    out.update(_extract_per_horizon_cols(rec))

    return out


def _rows_to_df(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame([])
    df = pd.DataFrame(rows)

    # numeric coercion for stable CSV export
    num_cols = [
        "hd_factor",
        "lambda_cons",
        "lambda_gw",
        "lambda_prior",
        "lambda_smooth",
        "lambda_mv",
        "r2",
        "mae",
        "mse",
        "rmse",
        "pss",
        "coverage80",
        "sharpness80",
        "epsilon_prior",
        "epsilon_cons",
        "epsilon_gw",
        "epsilon_cons_raw",
        "epsilon_gw_raw",
        "coverage80_uncal",
        "coverage80_cal",
        "sharpness80_uncal",
        "sharpness80_cal",
        "coverage80_uncal_phys",
        "coverage80_cal_phys",
        "sharpness80_uncal_phys",
        "sharpness80_cal_phys",
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # expanded per-horizon cols
    for c in list(df.columns):
        if c.startswith(("r2_H", "mae_H")):
            df[c] = pd.to_numeric(df[c], errors="coerce")
        if c.startswith(("coverage80_", "sharpness80_")):
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


def _to_long(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    base_cols = [
        "timestamp",
        "city",
        "model",
        "pde_mode",
        "lambda_cons",
        "lambda_gw",
        "lambda_prior",
        "lambda_smooth",
        "lambda_mv",
        "_src",
        "_run_dir",
    ]
    base_cols = [c for c in base_cols if c in df.columns]

    # discover horizons
    hs: List[str] = []
    for c in df.columns:
        if c.startswith("r2_H"):
            hs.append(c.replace("r2_", ""))
        if c.startswith("mae_H"):
            hs.append(c.replace("mae_", ""))
    hs = sorted(set(hs))

    if not hs:
        return pd.DataFrame([])

    rows: List[Dict[str, Any]] = []
    for _, r in df.iterrows():
        base = {k: r.get(k) for k in base_cols}
        for H in hs:
            rec: Dict[str, Any] = dict(base)
            rec["horizon"] = H
            rec["r2"] = r.get(f"r2_{H}", np.nan)
            rec["mae"] = r.get(f"mae_{H}", np.nan)

            rec["coverage80_uncal"] = r.get(
                f"coverage80_uncal_{H}", np.nan
            )
            rec["coverage80_cal"] = r.get(
                f"coverage80_cal_{H}", np.nan
            )
            rec["sharpness80_uncal"] = r.get(
                f"sharpness80_uncal_{H}", np.nan
            )
            rec["sharpness80_cal"] = r.get(
                f"sharpness80_cal_{H}", np.nan
            )
            rows.append(rec)

    out = pd.DataFrame(rows)
    for c in ["r2", "mae", "coverage80_uncal", "coverage80_cal"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    for c in ["sharpness80_uncal", "sharpness80_cal"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    return out


# ---------------------------------------------------------------------
# Filters + output
# ---------------------------------------------------------------------

def _filter_rows(
    rows: List[Dict[str, Any]],
    args: argparse.Namespace,
) -> List[Dict[str, Any]]:
    if not rows:
        return rows

    keep_cities = utils.resolve_cities(args)

    raw = str(args.city or "").strip()
    if raw:
        for p in raw.split(","):
            p = p.strip()
            if p:
                keep_cities.append(utils.canonical_city(p))

    keep_cities = [c for c in keep_cities if c]
    keep_cities = sorted(set(keep_cities))

    keep_models: List[str] = []
    raw_m = str(args.models or "").strip()
    if raw_m:
        keep_models = [
            m.strip() for m in raw_m.split(",") if m.strip()
        ]

    out: List[Dict[str, Any]] = []
    for r in rows:
        city = utils.canonical_city(str(r.get("city") or ""))
        model = str(r.get("model") or "")

        if keep_cities and city not in keep_cities:
            continue
        if keep_models and model not in keep_models:
            continue

        out.append(r)

    return out


def _resolve_out(*, out: str, out_dir: Optional[str]) -> Path:
    """Like resolve_fig_out(), but targets scripts/out/ by default."""
    p = Path(out).expanduser().with_suffix("")
    if out_dir:
        base = Path(out_dir).expanduser()
        return (base / p.name).resolve()
    if not p.is_absolute():
        p = cfg.OUT_DIR / p
    return p


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def build_model_metrics_main(argv: Optional[List[str]] = None) -> None:
    args = _parse_args(argv)
    utils.ensure_script_dirs()

    src = utils.as_path(args.src)
    rows = _scan_ablation_jsonl(src)
    rows = _filter_rows(rows, args)

    if utils.str_to_bool(args.dedupe, default=True):
        rows = _dedupe_records(rows)
    else:
        rows = [_scale_legacy_to_mm(r) for r in rows]

    if not rows:
        raise SystemExit(f"No ablation records found under: {src}")

    flat = [_record_to_row(r) for r in rows]
    df = _rows_to_df(flat)

    out_stem = _resolve_out(out=args.out, out_dir=args.out_dir)

    csv_p = out_stem.with_suffix(".csv")
    json_p = out_stem.with_suffix(".json")

    df.to_csv(csv_p, index=False)
    json_p.write_text(
        json.dumps(
            df.to_dict("records"),
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"[OK] wrote: {csv_p}")
    print(f"[OK] wrote: {json_p}")

    if utils.str_to_bool(args.include_long, default=True):
        df_long = _to_long(df)
        if not df_long.empty:
            csv_l = out_stem.with_name(out_stem.name + "_long")
            csv_l = csv_l.with_suffix(".csv")

            json_l = out_stem.with_name(out_stem.name + "_long")
            json_l = json_l.with_suffix(".json")

            df_long.to_csv(csv_l, index=False)
            json_l.write_text(
                json.dumps(
                    df_long.to_dict("records"),
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )

            print(f"[OK] wrote: {csv_l}")
            print(f"[OK] wrote: {json_l}")


def main(argv: Optional[List[str]] = None) -> None:
    build_model_metrics_main(argv)


if __name__ == "__main__":
    main()
