# geoprior/ui/map/selection_worker.py
# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import pandas as pd
from PyQt5.QtCore import (
    QObject,
    QRunnable,
    QThreadPool,
    pyqtSignal,
)

from .selection_stats import (
    group_trend,
    load_series_for_ids,
    summarize_series,
    exceed_prob_from_quantiles,
    pick_mid_col,
    band_cols,
)


_Q_RE = re.compile(r"(?:^|_)q(\d{1,2})$", re.I)
_BASE_STRIP = (
    "_actual",
    "_observed",
    "_obs",
    "_true",
    "_target",
)


def _strip_base(z: str) -> str:
    zz = str(z or "").strip()
    low = zz.lower()
    for suf in _BASE_STRIP:
        if low.endswith(suf):
            return zz[: -len(suf)]
    return zz


def _infer_q_family(
    z: str,
    cols: Sequence[str],
) -> list[str]:
    # 1) if z is already a quantile col, keep old logic
    out = _q_family(z, cols)
    if out:
        return out

    # 2) try base from *_actual, *_obs, ...
    base = _strip_base(z)
    base = str(base or "").strip()
    if not base:
        return []

    probe = f"{base}_q50"
    return _q_family(probe, cols)


def _q_meta(cols: Sequence[str]) -> list[Tuple[float, str]]:
    out: list[Tuple[float, str]] = []
    for c in (cols if cols is not None else []):
        m = _Q_RE.search(str(c))
        if not m:
            continue
        try:
            p = float(int(m.group(1))) / 100.0
        except Exception:
            continue
        out.append((p, str(c)))
    out.sort(key=lambda it: it[0])
    return out

def _is_quantile_col(name: str) -> bool:
    return bool(_Q_RE.search(str(name or ""))) 

def _q_family(
    z: str,
    cols: Sequence[str],
) -> list[str]:
    """
    Try to keep quantiles from the same family as z.

    Example:
    z = "subsidence_q50"
    -> family prefix "subsidence_"
    -> picks subsidence_q05, subsidence_q95, ...
    """
    z = str(z or "").strip()
    m = re.match(r"^(.*?)(?:_)?q\d{1,2}$", z, re.I)
    pref = str(m.group(1)) if m else ""
    if pref and (not pref.endswith("_")):
        pref = pref + "_"

    out: list[str] = []
    for c in (cols if cols is not None else []):
        cc = str(c)
        if pref and (not cc.startswith(pref)):
            continue
        if _Q_RE.search(cc):
            out.append(cc)
    return out


@dataclass(frozen=True)
class SelectionRequest:
    mode: str
    ids: Sequence[int]

    id_col: str
    t_col: str
    z_col: str

    # Optional sources
    df_all: Optional[pd.DataFrame] = None
    df_frame: Optional[pd.DataFrame] = None
    path: Optional[Path] = None

    # Risk threshold (optional)
    thr: Optional[float] = None


class _Signals(QObject):
    result = pyqtSignal(int, object)
    error = pyqtSignal(int, str)
    done = pyqtSignal(int)


class _Job(QRunnable):
    def __init__(
        self,
        token: int,
        req: SelectionRequest,
    ) -> None:
        super().__init__()
        self.token = int(token)
        self.req = req
        self.signals = _Signals()

    def run(self) -> None:
        try:
            res = _compute(self.req)
            self.signals.result.emit(self.token, res)
        except Exception as e:
            self.signals.error.emit(self.token, str(e))
        finally:
            self.signals.done.emit(self.token)


def _compute(req: SelectionRequest) -> Dict[str, Any]:
    mode = str(req.mode or "off").strip().lower()
    ids = [int(i) for i in (req.ids or [])]

    if mode not in ("point", "group"):
        return {"mode": "off", "ids": []}

    if not ids:
        return {"mode": mode, "ids": [], "empty": True}

    # -------------------------
    # Prefer brain df_all
    # -------------------------
    df = None
    if (
        req.df_all is not None
        and isinstance(req.df_all, pd.DataFrame)
        and (not req.df_all.empty)
        and (req.id_col in req.df_all.columns)
    ):
        cols = [req.id_col, req.t_col, req.z_col]
        cols = [c for c in cols if c in req.df_all.columns]

        # qcols = _q_family(req.z_col, req.df_all.columns)
        qcols = _infer_q_family(
            req.z_col,
            req.df_all.columns,
        )
        for qc in qcols:
            if qc not in cols:
                cols.append(qc)

        m = req.df_all[req.id_col].isin(ids)
        df = req.df_all.loc[m, cols].copy()
        df["_pid"] = df[req.id_col].astype(str)

    # -------------------------
    # Fallback: read CSV by ids
    # -------------------------
    if df is None:
        if req.path is None:
            return {"mode": mode, "ids": ids, "empty": True}

        p = Path(req.path)
        qcols: list[str] = []
        try:
            hdr = pd.read_csv(p, nrows=0).columns
            qcols = _q_family(req.z_col, hdr)
        except:
            qcols = []
            
        df = load_series_for_ids(
            path=p,
            id_col=req.id_col,
            ids=ids,
            t=req.t_col,
            z=req.z_col,
            q_cols=qcols,
        )

    if df is None or df.empty:
        return {"mode": mode, "ids": ids, "empty": True}

    # -------------------------
    # Summary
    # -------------------------
    summ = summarize_series(df, t=req.t_col, z=req.z_col)

    out: Dict[str, Any] = {
        "mode": mode,
        "ids": ids,
        "id_col": str(req.id_col),
        "t_col": str(req.t_col),
        "z_col": str(req.z_col),
        "summary": {
            "n_points": summ.n_points,
            "t_min": summ.t_min,
            "t_max": summ.t_max,
            "z_min": summ.z_min,
            "z_max": summ.z_max,
            "z_mean": summ.z_mean,
        },
    }

    # -------------------------
    # Quantile metadata (if any)
    # -------------------------
    qcols = _infer_q_family(req.z_col, df.columns)
    qmeta = _q_meta(qcols)
    mid = pick_mid_col(req.z_col, qmeta)

    # If user selected *_actual (not a quantile col),
    # keep mid as actual. Only auto-pick q50 when
    # z_col itself is from the quantile family.
    mid = str(req.z_col)
    if _Q_RE.search(str(req.z_col or "")):
        mid = pick_mid_col(req.z_col, qmeta)

    lo = hi = None
    if qmeta:
        lo, hi = band_cols(qmeta, band="80")

    lo, hi = (None, None)
    if qmeta:
        lo, hi = band_cols(qmeta, band="80")
    # -------------------------
    # Point vs group outputs
    # -------------------------
    if mode == "point":
        pid = ids[0]
        d1 = df[df[req.id_col] == pid].copy()
        if req.t_col in d1.columns:
            d1 = d1.sort_values(req.t_col, kind="mergesort")
        out["series"] = d1
        
        if lo and hi:
            out["band"] = (str(lo), str(hi))
        # UX: if user plots "actual", overlay q50
        
        overlay = None
        if qmeta and (not _is_quantile_col(req.z_col)):
            q50 = pick_mid_col(req.z_col, qmeta)
            if q50 and (q50 != req.z_col):
                if q50 in d1.columns:
                    overlay = str(q50)
        out["overlay"] = overlay

        if req.thr is not None and qmeta:
            try:
                row = d1.iloc[-1]
                out["risk_p"] = float(
                    exceed_prob_from_quantiles(
                        row,
                        thr=float(req.thr),
                        q_cols=qmeta,
                    )
                )
            except Exception:
                out["risk_p"] = float("nan")
                


        return out

    # group mode
    tr = group_trend(df, t=req.t_col, mid=mid)
    out["trend"] = tr

    cur = None
    if (
        req.df_frame is not None
        and isinstance(req.df_frame, pd.DataFrame)
        and (not req.df_frame.empty)
        and (req.id_col in req.df_frame.columns)
        and (req.z_col in req.df_frame.columns)
    ):
        s = req.df_frame.loc[
            req.df_frame[req.id_col].isin(ids),
            req.z_col,
        ]
        s = pd.to_numeric(s, errors="coerce").dropna()
        if not s.empty:
            cur = {
                "n": int(s.shape[0]),
                "min": float(s.min()),
                "max": float(s.max()),
                "mean": float(s.mean()),
                "median": float(s.median()),
            }
    out["current"] = cur
    return out


class SelectionWorker(QObject):
    """
    Threaded selection analytics.

    Use:
      worker.request(req)

    Emits:
      result_ready(dict)
      error(str)

    Notes
    -----
    - Stale jobs are ignored by token.
    - Compute prefers df_all from the brain.
    """

    result_ready = pyqtSignal(object)
    error = pyqtSignal(str)
    busy_changed = pyqtSignal(bool)

    def __init__(
        self,
        *,
        parent: Optional[QObject] = None,
    ) -> None:
        super().__init__(parent)
        self._pool = QThreadPool.globalInstance()
        self._tok = 0
        
    def cancel(self) -> None:
        self._tok += 1
        self.busy_changed.emit(False)

    def request(self, req: SelectionRequest) -> None:
        self._tok += 1
        tok = int(self._tok)

        self.busy_changed.emit(True)

        job = _Job(tok, req)

        job.signals.result.connect(self._on_result)
        job.signals.error.connect(self._on_error)
        job.signals.done.connect(self._on_done)

        self._pool.start(job)

    def _on_result(self, tok: int, res: object) -> None:
        if int(tok) != int(self._tok):
            return
        self.result_ready.emit(res)

    def _on_error(self, tok: int, msg: str) -> None:
        if int(tok) != int(self._tok):
            return
        self.error.emit(str(msg))

    def _on_done(self, tok: int) -> None:
        if int(tok) != int(self._tok):
            return
        self.busy_changed.emit(False)

