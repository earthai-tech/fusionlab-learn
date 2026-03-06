# scripts/rebuild_confusion_tables.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd


@dataclass(frozen=True)
class RowSpec:
    name: str
    acc: float
    f1: float
    prec: float
    rec: float


def _safe_float(x: object) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, float) and math.isnan(x):
        return None
    try:
        return float(x)
    except Exception:
        return None


def _round2(x: float) -> float:
    return float(f"{x:.2f}")


def _acc(tp: int, tn: int, fp: int, fn: int) -> float:
    den = tp + tn + fp + fn
    return 100.0 * (tp + tn) / den if den else 0.0


def _prec(tp: int, fp: int) -> Optional[float]:
    den = tp + fp
    return 100.0 * tp / den if den else None


def _rec(tp: int, fn: int) -> Optional[float]:
    den = tp + fn
    return 100.0 * tp / den if den else None


def _f1(tp: int, fp: int, fn: int) -> Optional[float]:
    den = 2 * tp + fp + fn
    return 100.0 * (2 * tp) / den if den else None


def _spec(tn: int, fp: int) -> Optional[float]:
    den = tn + fp
    return 100.0 * tn / den if den else None


def _npv(tn: int, fn: int) -> Optional[float]:
    den = tn + fn
    return 100.0 * tn / den if den else None


def _mcc(tp: int, tn: int, fp: int, fn: int) -> float:
    a = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    if a <= 0:
        return 0.0
    num = tp * tn - fp * fn
    return num / math.sqrt(a)


def _close(a: Optional[float], b: float, tol: float) -> bool:
    if a is None:
        return False
    return abs(a - b) <= tol


def _reconstruct_counts(
    *,
    p: int,
    n: int,
    acc: float,
    prec: float,
    rec: float,
    f1: float,
    tol: float,
) -> Optional[Tuple[int, int, int, int]]:
    """
    Reconstruct integer TP/FP/FN/TN given (P,N) and
    reported metrics (Acc, Prec, Rec, F1).

    Strategy:
    - Enumerate TP candidates consistent with Recall.
    - For each TP, enumerate FP candidates.
    - Derive FN, TN from P,N.
    - Validate all metrics within tolerance.
    """
    # Candidate TP around Recall * P
    tp0 = int(round((rec / 100.0) * p))
    tp_cands = set([tp0 - 2, tp0 - 1, tp0, tp0 + 1, tp0 + 2])

    for tp in sorted(tp_cands):
        if tp < 0 or tp > p:
            continue
        fn = p - tp

        # From precision: prec = TP/(TP+FP)
        # => FP = TP*(100/prec - 1)
        fp0 = int(round(tp * (100.0 / prec - 1.0)))
        fp_cands = set(
            [fp0 - 3, fp0 - 2, fp0 - 1, fp0, fp0 + 1, fp0 + 2, fp0 + 3]
        )

        for fp in sorted(fp_cands):
            if fp < 0 or fp > n:
                continue
            tn = n - fp

            a = _round2(_acc(tp, tn, fp, fn))
            pr = _prec(tp, fp)
            rc = _rec(tp, fn)
            f1v = _f1(tp, fp, fn)

            if not _close(a, acc, tol):
                continue
            if not _close(_round2(pr or -1.0), prec, tol):
                continue
            if not _close(_round2(rc or -1.0), rec, tol):
                continue
            if not _close(_round2(f1v or -1.0), f1, tol):
                continue

            return tp, fp, fn, tn

    return None


def _infer_pn(
    rows: List[RowSpec],
    *,
    n_test: int,
    tol: float,
) -> Tuple[int, int]:
    """
    Find (P,N) such that every row can be reconstructed.
    """
    best: Tuple[int, int, int] = (-1, -1, -1)
    for p in range(0, n_test + 1):
        n = n_test - p
        ok = 0
        for r in rows:
            out = _reconstruct_counts(
                p=p,
                n=n,
                acc=r.acc,
                prec=r.prec,
                rec=r.rec,
                f1=r.f1,
                tol=tol,
            )
            if out is not None:
                ok += 1
        if ok > best[2]:
            best = (p, n, ok)
        if ok == len(rows):
            return p, n

    p, n, ok = best
    raise RuntimeError(
        "Could not find a single (P,N) that "
        "matches all rows. Best match: "
        f"P={p}, N={n}, rows_ok={ok}/{len(rows)}. "
        "Try increasing --tol."
    )


def _load_metric_rows(csv_path: Path) -> List[RowSpec]:
    df = pd.read_csv(csv_path)
    need = ["Model", "Acc", "F1", "Prec", "Rec"]
    for c in need:
        if c not in df.columns:
            raise ValueError(f"Missing column: {c}")

    rows: List[RowSpec] = []
    for _, r in df.iterrows():
        rows.append(
            RowSpec(
                name=str(r["Model"]),
                acc=float(r["Acc"]),
                f1=float(r["F1"]),
                prec=float(r["Prec"]),
                rec=float(r["Rec"]),
            )
        )
    return rows


def _counts_to_df(
    rows: List[RowSpec],
    *,
    p: int,
    n: int,
    tol: float,
) -> pd.DataFrame:
    out_rows: List[Dict[str, object]] = []
    for r in rows:
        counts = _reconstruct_counts(
            p=p,
            n=n,
            acc=r.acc,
            prec=r.prec,
            rec=r.rec,
            f1=r.f1,
            tol=tol,
        )
        if counts is None:
            raise RuntimeError(
                f"Failed reconstruction for {r.name}. "
                "Increase --tol or check inputs."
            )

        tp, fp, fn, tn = counts
        out_rows.append(
            {
                "Model": r.name,
                "TP": tp,
                "FP": fp,
                "FN": fn,
                "TN": tn,
                "Acc%": _round2(_acc(tp, tn, fp, fn)),
                "Prec%": _round2(_prec(tp, fp) or float("nan")),
                "Rec%": _round2(_rec(tp, fn) or float("nan")),
                "F1%": _round2(_f1(tp, fp, fn) or float("nan")),
                "Specificity%": _round2(_spec(tn, fp) or float("nan")),
                "NPV%": _round2(_npv(tn, fn) or float("nan")),
                "MCC": float(f"{_mcc(tp, tn, fp, fn):.4f}"),
                "PredPos": tp + fp,
                "PredNeg": tn + fn,
            }
        )

    df = pd.DataFrame(out_rows)
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics-csv", required=True)
    ap.add_argument("--data-csv", default=None)
    ap.add_argument("--target", default="cath")
    ap.add_argument("--test-ratio", type=float, default=0.30)
    ap.add_argument("--n-test", type=int, default=None)
    ap.add_argument("--tol", type=float, default=0.02)
    ap.add_argument("--outdir", default="scripts/out")
    ap.add_argument("--tag", default="table")

    args = ap.parse_args()

    rows = _load_metric_rows(Path(args.metrics_csv))

    # Determine n_test
    if args.n_test is not None:
        n_test = int(args.n_test)
    elif args.data_csv is not None:
        df_data = pd.read_csv(args.data_csv)
        n_total = int(df_data.shape[0])
        n_test = int(round(n_total * float(args.test_ratio)))
    else:
        raise ValueError(
            "Provide --n-test or --data-csv."
        )

    p, n = _infer_pn(rows, n_test=n_test, tol=float(args.tol))

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = _counts_to_df(rows, p=p, n=n, tol=float(args.tol))

    # Full export
    full_path = outdir / f"{args.tag}_confusion_full.csv"
    df.to_csv(full_path, index=False)

    # Paper-ready export (rename Rec% as Recall%)
    paper = df[
        [
            "Model",
            "Acc%",
            "Prec%",
            "Rec%",
            "Specificity%",
            "NPV%",
            "F1%",
            "MCC",
        ]
    ].copy()
    paper = paper.rename(columns={"Rec%": "Recall%"})
    paper_path = outdir / f"{args.tag}_metrics_paper.csv"
    paper.to_csv(paper_path, index=False)

    # Print summary
    print(f"n_test={n_test}, P={p}, N={n}")
    print(f"Wrote: {full_path}")
    print(f"Wrote: {paper_path}")


if __name__ == "__main__":
    main()