# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>

"""
Collect SM3 per-regime summaries into one combined table.

Expected structure under --suite-root:
  sm3_tau_<reg>_50/
    sm3_synth_summary.csv

Writes:
  --out-csv  combined CSV (long format)
  --out-json combined JSON (records)

Example:
  python nat.com/sm3_collect_summaries.py \
    --suite-root results/sm3_tau_suite_20260303-120000 \
    --out-csv results/.../combined.csv \
    --out-json results/.../combined.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import pandas as pd


def infer_regime(folder_name: str) -> str:
    # Matches:
    #   sm3_tau_<reg>_50
    #   sm3_both_<reg>_50
    m = re.search(r"sm3_(?:tau|both)_(.+?)_50$", folder_name)
    if m:
        return m.group(1)
    return folder_name


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite-root", required=True)
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--out-json", required=True)
    args = ap.parse_args()

    root = Path(args.suite_root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Suite root not found: {root}")

    rows = []

    # Scan for summary CSVs anywhere under suite root
    for p in root.rglob("sm3_synth_summary.csv"):
        run_dir = p.parent
        regime = infer_regime(run_dir.name)

        try:
            df = pd.read_csv(p)
        except Exception as e:
            print(f"[skip] failed to read {p}: {e}")
            continue

        if df.empty or "metric" not in df.columns:
            print(f"[skip] unexpected format: {p}")
            continue

        df = df.copy()
        df.insert(0, "regime", regime)
        df.insert(1, "run_dir", str(run_dir))
        rows.append(df)

    if not rows:
        raise RuntimeError(
            "No sm3_synth_summary.csv files found under suite root."
        )

    out = pd.concat(rows, ignore_index=True)

    # Sort for readability
    sort_cols = [c for c in ["metric", "regime"] if c in out.columns]
    if sort_cols:
        out = out.sort_values(sort_cols).reset_index(drop=True)

    out_csv = Path(args.out_csv).expanduser().resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)

    out_json = Path(args.out_json).expanduser().resolve()
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(out.to_dict("records"), f, indent=2)

    print("[OK] wrote:", str(out_csv))
    print("[OK] wrote:", str(out_json))


if __name__ == "__main__":
    main()