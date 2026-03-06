# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from fusionlab.utils.transfer import xfer_io
from fusionlab.utils.transfer import xfer_metrics


def _unit_scale(df: pd.DataFrame) -> float:
    if "subsidence_unit" not in df.columns:
        return 1.0
    s = df["subsidence_unit"].dropna()
    if s.empty:
        return 1.0
    u0 = str(s.astype(str).iloc[0]).strip().lower()
    if u0.startswith("mm"):
        return 1.0 / 1000.0
    return 1.0


def _is_missing(x) -> bool:
    if x is None:
        return True
    try:
        return pd.isna(x)
    except Exception:
        return False


def fix_one_run(run_dir: Path) -> bool:
    csv_p = run_dir / "xfer_results.csv"
    js_p = run_dir / "xfer_results.json"
    if not (csv_p.exists() and js_p.exists()):
        return False

    df = pd.read_csv(csv_p)
    rows = json.loads(js_p.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        return False

    changed = False

    for r in rows:
        csv_eval = r.get("csv_eval")
        if not csv_eval:
            continue

        p = Path(str(csv_eval))
        if not p.is_absolute():
            # repo root = .../fusionlab-learn
            repo = Path(__file__).resolve().parents[1]
            p = repo / p

        if not p.exists():
            continue

        try:
            ev = pd.read_csv(p)
        except Exception:
            continue

        summ = xfer_metrics.summarize_eval_df(ev)
        sc = _unit_scale(ev)

        cov = float(summ.coverage80)
        shp = float(summ.sharpness80) * sc

        if _is_missing(r.get("coverage80")):
            r["coverage80"] = cov
            changed = True
        if _is_missing(r.get("sharpness80")):
            r["sharpness80"] = shp
            changed = True

        # also patch the CSV row
        m = df["strategy"].astype(str).str.lower().eq(
            str(r.get("strategy", "")).lower()
        )
        m &= df["direction"].astype(str).eq(
            str(r.get("direction", ""))
        )
        m &= df["split"].astype(str).str.lower().eq(
            str(r.get("split", "")).lower()
        )
        m &= df["calibration"].astype(str).str.lower().eq(
            str(r.get("calibration", "")).lower()
        )

        if m.any():
            if "coverage80" in df.columns and (
                _is_missing(df.loc[m, "coverage80"].iloc[0])
            ):
                df.loc[m, "coverage80"] = cov
                changed = True
            if "sharpness80" in df.columns and (
                _is_missing(df.loc[m, "sharpness80"].iloc[0])
            ):
                df.loc[m, "sharpness80"] = shp
                changed = True

    if not changed:
        return False

    csv_p.write_text(df.to_csv(index=False), encoding="utf-8")
    js_p.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="results/xfer")
    args = ap.parse_args()

    src = Path(args.src).expanduser()
    if not src.exists():
        raise SystemExit(f"Missing: {src}")

    run_dirs = []
    if (src / "xfer_results.csv").exists():
        run_dirs = [src]
    else:
        for p in src.rglob("xfer_results.csv"):
            run_dirs.append(p.parent)

    n_ok = 0
    for rd in sorted(set(run_dirs)):
        if fix_one_run(rd):
            print(f"[fix] {rd}")
            n_ok += 1

    print(f"[OK] updated {n_ok} run dirs")


if __name__ == "__main__":
    main()