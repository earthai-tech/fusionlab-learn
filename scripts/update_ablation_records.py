# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio
#
# scripts/update_ablation_records.py
#
"""Update ablation_record.jsonl using post-hoc calibrated metrics.

Why this script
---------------
Older ablation JSONL rows may store point/interval metrics in
SI units (e.g., meters) or store *uncalibrated* interval metrics.

This script patches each JSONL row by reading the authoritative
run artifacts:

- geoprior_eval_phys_<TS>_interpretable.json  (preferred)
- geoprior_eval_phys_<TS>.json                (fallback)
- *eval_diagnostics*_calibrated.json          (fallback)

Outputs
-------
For each input ablation JSONL:
- writes a sibling file ``ablation_record.updated.jsonl`` next
  to the original (does not overwrite by default).

Optionally also exports a combined table (CSV + JSON) under
``scripts/out``.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from . import config as cfg
from . import utils


_PHYS_PREFIX = "geoprior_eval_phys_"
_PHYS_EXT = ".json"


@dataclass(frozen=True)
class PatchInfo:
    ok: bool
    reason: str = ""
    ts: str = ""
    run_dir: str = ""
    phys_json: str = ""
    eval_diag: str = ""


def _parse_args(
    argv: Optional[List[str]],
) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="update-ablation-records",
        description=(
            "Patch ablation_record.jsonl to use post-hoc "
            "calibrated metrics in interpretable units."
        ),
    )

    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument(
        "--results-root",
        type=str,
        default=None,
        help=(
            "Scan recursively for run folders + ablation "
            "records under this root."
        ),
    )
    g.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help=(
            "Update only the ablation record(s) found under "
            "this run directory."
        ),
    )

    p.add_argument(
        "--ablation",
        type=str,
        default=None,
        help=(
            "Optional explicit path to a JSONL ablation record. "
            "If provided, only this file is processed."
        ),
    )

    p.add_argument(
        "--include-updated-inputs",
        type=str,
        default="false",
        help=(
            "Also process ablation_record.updated*.jsonl as "
            "inputs (true/false)."
        ),
    )

    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing *.updated.jsonl outputs.",
    )
    p.add_argument(
        "--backup",
        action="store_true",
        help="Write a .bak copy of the input JSONL.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not write; just report.",
    )

    p.add_argument(
        "--no-table",
        action="store_true",
        help="Skip exporting the combined CSV/JSON table.",
    )
    p.add_argument(
        "--table-stem",
        type=str,
        default="ablation_records_table.updated",
        help=(
            "Output stem for the combined table under scripts/out/. "
            "Writes <stem>.csv and <stem>.json"
        ),
    )

    p.add_argument(
        "--verbose",
        type=str,
        default="true",
        help="Print per-file status (true/false).",
    )

    return p.parse_args(argv)


def _is_updated_name(p: Path) -> bool:
    return "updated" in p.name.lower()


def _out_updated_path(src: Path) -> Path:
    if src.name == "ablation_record.jsonl":
        return src.with_name("ablation_record.updated.jsonl")

    stem = src.stem
    if stem.endswith(".updated"):
        return src

    return src.with_name(stem + ".updated" + src.suffix)


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for ln in path.read_text(encoding="utf-8").splitlines():
        s = ln.strip()
        if not s:
            continue
        try:
            obj = json.loads(s)
        except Exception:
            continue
        if isinstance(obj, dict):
            rows.append(obj)
    return rows


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    out = [json.dumps(r, ensure_ascii=False) for r in rows]
    path.write_text("\n".join(out) + "\n", encoding="utf-8")


def _safe_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        v = float(x)
    except Exception:
        return None
    if not math.isfinite(v):
        return None
    return v


def _parse_ts_from_phys_json(name: str) -> Optional[str]:
    if not (name.startswith(_PHYS_PREFIX) and name.endswith(_PHYS_EXT)):
        return None

    core = name[len(_PHYS_PREFIX) : -len(_PHYS_EXT)]
    if core.endswith("_interpretable"):
        core = core[: -len("_interpretable")]

    return core or None


def _build_ts_index(root: Path) -> Dict[str, Path]:
    """Map <timestamp> -> run_dir by scanning phys JSONs."""
    idx: Dict[str, Tuple[int, float, Path]] = {}

    for fp in root.rglob(f"{_PHYS_PREFIX}*{_PHYS_EXT}"):
        ts = _parse_ts_from_phys_json(fp.name)
        if not ts:
            continue

        is_interp = int("interpretable" in fp.name)
        mtime = fp.stat().st_mtime
        cur = idx.get(ts)

        if cur is None:
            idx[ts] = (is_interp, mtime, fp.parent)
            continue

        if (is_interp, mtime) > (cur[0], cur[1]):
            idx[ts] = (is_interp, mtime, fp.parent)

    return {k: v[2] for k, v in idx.items()}


def _find_phys_json_for_ts(
    run_dir: Path,
    ts: str,
) -> Optional[Path]:
    p1 = run_dir / f"{_PHYS_PREFIX}{ts}_interpretable.json"
    if p1.exists():
        return p1

    p2 = run_dir / f"{_PHYS_PREFIX}{ts}.json"
    if p2.exists():
        return p2

    # last resort: newest preferred under this run_dir
    return utils.find_phys_json(run_dir)


def _infer_run_dir_from_ablation(
    ab_path: Path,
    ts: str,
) -> Optional[Path]:
    base = ab_path.parent
    if base.name == "ablation_records":
        base = base.parent

    for nm in [
        f"{_PHYS_PREFIX}{ts}_interpretable.json",
        f"{_PHYS_PREFIX}{ts}.json",
    ]:
        cand = base / nm
        if cand.exists():
            return cand.parent

    # fallback: search within base
    for nm in [
        f"{_PHYS_PREFIX}{ts}_interpretable.json",
        f"{_PHYS_PREFIX}{ts}.json",
    ]:
        hits = list(base.rglob(nm))
        if hits:
            hits.sort(
                key=lambda x: x.stat().st_mtime,
                reverse=True,
            )
            return hits[0].parent

    return None


def _pick_diag_path(run_dir: Path) -> Optional[Path]:
    return utils.find_eval_diag_json(run_dir)


def _patch_record(
    rec: Dict[str, Any],
    *,
    ab_path: Path,
    ts_index: Dict[str, Path],
) -> Tuple[Dict[str, Any], PatchInfo]:
    ts = str(rec.get("timestamp") or "").strip()
    if not ts:
        return rec, PatchInfo(ok=False, reason="no_timestamp")

    run_dir = ts_index.get(ts)
    if run_dir is None:
        run_dir = _infer_run_dir_from_ablation(ab_path, ts)

    if run_dir is None:
        return rec, PatchInfo(
            ok=False,
            ts=ts,
            reason="run_dir_not_found",
        )

    phys_p = _find_phys_json_for_ts(run_dir, ts)
    if phys_p is None or not phys_p.exists():
        return rec, PatchInfo(
            ok=False,
            ts=ts,
            run_dir=str(run_dir),
            reason="phys_json_missing",
        )

    diag_p = _pick_diag_path(run_dir)

    phys = utils.safe_load_json(phys_p)
    phys = utils.phys_json_to_mm(phys)

    diag = utils.safe_load_json(diag_p) if diag_p else {}

    r2, mae, mse = utils.pick_point_metrics(phys, diag)
    cov, shp = utils.pick_interval_metrics(phys, diag)

    pm = phys.get("point_metrics") or {}
    rmse = _safe_float(pm.get("rmse"))
    if rmse is None and _safe_float(mse) is not None:
        rmse = math.sqrt(float(mse))

    # preserve old headline metrics
    legacy: Dict[str, Any] = {}
    for k in [
        "r2",
        "mse",
        "mae",
        "rmse",
        "coverage80",
        "sharpness80",
    ]:
        if k in rec:
            legacy[k] = rec.get(k)
    rec.setdefault("legacy", {})
    rec["legacy"]["pre_posthoc_patch"] = legacy

    units = phys.get("units") or {}
    rec["units"] = units

    # headline metrics (paper-consistent)
    rec["r2"] = _safe_float(r2)
    rec["mse"] = _safe_float(mse)
    rec["mae"] = _safe_float(mae)
    rec["rmse"] = _safe_float(rmse)
    rec["coverage80"] = _safe_float(cov)
    rec["sharpness80"] = _safe_float(shp)

    # physics diagnostics (if present)
    phys_diag = phys.get("physics_diagnostics") or {}
    for k in ["epsilon_prior", "epsilon_cons", "epsilon_gw"]:
        v = phys_diag.get(k)
        if v is not None:
            rec[k] = _safe_float(v)

    # per-horizon (keep whole block for downstream scripts)
    if "per_horizon" in phys:
        rec["per_horizon"] = phys.get("per_horizon")

    # optional: bring PSS from diagnostics when present
    pss = None
    blk = diag.get("__overall__") if isinstance(diag, dict) else None
    if isinstance(blk, dict):
        pss = blk.get("pss")
    if pss is not None:
        rec["pss"] = _safe_float(pss)

    # traceability blocks
    rec["metrics"] = {
        "posthoc": {
            "point_metrics": phys.get("point_metrics") or {},
            "interval_calibration": (
                phys.get("interval_calibration") or {}
            ),
            "per_horizon": phys.get("per_horizon") or {},
        },
        "raw": {
            "metrics_evaluate": (
                phys.get("metrics_evaluate") or {}
            ),
            "eval_diag_flat": utils.flatten_eval_diag(diag),
        },
        "sources": {
            "phys_json": str(phys_p),
            "eval_diag_json": (
                str(diag_p) if diag_p else None
            ),
            "run_dir": str(run_dir),
        },
    }

    return rec, PatchInfo(
        ok=True,
        ts=ts,
        run_dir=str(run_dir),
        phys_json=str(phys_p),
        eval_diag=str(diag_p) if diag_p else "",
    )


def _discover_ablation_files(
    *,
    root: Path,
    ablation: Optional[str],
    include_updated: bool,
) -> List[Path]:
    if ablation:
        p = Path(ablation).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(str(p))
        return [p]

    files = utils.find_all(
        root,
        cfg.PATTERNS.get("ablation_record_jsonl", ()),
        must_exist=False,
    )

    if not include_updated:
        base = [p for p in files if not _is_updated_name(p)]
        if base:
            return base

    return files


def _simple_row(rec: Dict[str, Any]) -> Dict[str, Any]:
    row: Dict[str, Any] = {}
    for k, v in rec.items():
        if isinstance(v, (str, int, float, bool)) or v is None:
            row[k] = v
    return row


def update_ablation_records_main(
    argv: Optional[List[str]] = None,
) -> None:
    args = _parse_args(argv)

    verbose = utils.str_to_bool(args.verbose, default=True)
    inc_upd = utils.str_to_bool(
        args.include_updated_inputs,
        default=False,
    )

    if args.results_root:
        root = Path(args.results_root).expanduser().resolve()
    else:
        root = Path(args.run_dir).expanduser().resolve()  # type: ignore[arg-type]

    if not root.exists():
        raise FileNotFoundError(str(root))

    utils.ensure_script_dirs()

    # Build <timestamp> -> run_dir map.
    ts_index = _build_ts_index(root)

    ab_files = _discover_ablation_files(
        root=root,
        ablation=args.ablation,
        include_updated=inc_upd,
    )

    if not ab_files:
        raise FileNotFoundError(
            f"No ablation_record*.jsonl found under {root}"
        )

    all_rows: List[Dict[str, Any]] = []

    for ab_path in ab_files:
        ab_path = ab_path.expanduser().resolve()
        rows = _read_jsonl(ab_path)
        if not rows:
            if verbose:
                print(f"[Skip] empty: {ab_path}")
            continue

        patched: List[Dict[str, Any]] = []
        infos: List[PatchInfo] = []

        for rec in rows:
            rec2, info = _patch_record(
                rec,
                ab_path=ab_path,
                ts_index=ts_index,
            )
            patched.append(rec2)
            infos.append(info)

        ok_n = sum(1 for i in infos if i.ok)
        bad = [i for i in infos if not i.ok]

        if verbose:
            print(
                f"[File] {ab_path.name}: patched {ok_n}/{len(infos)}"
            )
            for i in bad[:8]:
                print(
                    f"  - ts={i.ts} reason={i.reason}"
                )
            if len(bad) > 8:
                print(f"  ... +{len(bad) - 8} more")

        out_p = _out_updated_path(ab_path)

        if out_p.exists() and not args.overwrite:
            if verbose:
                print(
                    f"[Skip] exists (use --overwrite): {out_p}"
                )
        else:
            if args.backup:
                bak = ab_path.with_suffix(
                    ab_path.suffix + ".bak"
                )
                bak.write_text(
                    ab_path.read_text(encoding="utf-8"),
                    encoding="utf-8",
                )
                if verbose:
                    print(f"[OK] backup: {bak}")

            if not args.dry_run:
                _write_jsonl(out_p, patched)
                if verbose:
                    print(f"[OK] wrote: {out_p}")

        for r in patched:
            row = _simple_row(r)
            row["_src"] = str(ab_path)
            srcs = (r.get("metrics") or {}).get("sources") or {}
            if isinstance(srcs, dict):
                row["_run_dir"] = srcs.get("run_dir")
            all_rows.append(row)

    if args.no_table:
        return

    if not all_rows:
        return

    df = pd.DataFrame(all_rows)
    df = df.sort_values(
        by=["city", "timestamp"],
        na_position="last",
    )

    stem = str(args.table_stem).strip() or "ablation_table"
    out_csv = utils.resolve_out_out(stem + ".csv")
    out_js = utils.resolve_out_out(stem + ".json")

    if not args.dry_run:
        df.to_csv(out_csv, index=False)
        out_js.write_text(
            json.dumps(
                all_rows,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    if verbose:
        print(f"[OK] table: {out_csv}")
        print(f"[OK] table: {out_js}")

def main(argv: List[str] | None = None) -> None:
    update_ablation_records_main(argv)
    
if __name__ == "__main__":
    main()
