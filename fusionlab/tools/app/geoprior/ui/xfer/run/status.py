# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>

"""
Store-driven checklist statuses for Xfer RUN mode.

Notes
-----
- This mirrors inference/tune:
  store-only checks, no filesystem validation.
- Runtime/path checks belong in preview panels.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional

from ....config.store import GeoConfigStore

__all__ = ["compute_xfer_nav"]


def _ok(txt: str = "OK") -> Dict[str, str]:
    return {"status": "ok", "text": txt}


def _warn(txt: str = "Fix") -> Dict[str, str]:
    return {"status": "warn", "text": txt}


def _err(txt: str = "Err") -> Dict[str, str]:
    return {"status": "err", "text": txt}


def _get(store: GeoConfigStore, key: str, default: Any) -> Any:
    try:
        return store.get(key, default)
    except Exception:
        return default


def _as_str(x: Any) -> str:
    return str(x or "").strip()


def _as_list(x: Any) -> list:
    if x is None:
        return []
    if isinstance(x, (list, tuple, set)):
        return list(x)
    return []


def _valid_opt_bool(x: Any) -> bool:
    return (x is None) or (x is True) or (x is False)


def _valid_quantiles(x: Any) -> bool:
    if x is None:
        return True
    if not isinstance(x, (list, tuple)):
        return False
    if len(x) == 0:
        return False
    try:
        qq = [float(v) for v in x]
    except Exception:
        return False
    for q in qq:
        if not (0.0 < q < 1.0):
            return False
    return True


def compute_xfer_nav(store: GeoConfigStore) -> Dict[str, Dict[str, str]]:
    """
    Keys
    ----
    - cities
    - outputs
    - strategy
    - results
    """
    out: Dict[str, Dict[str, str]] = {}

    # -------------------------
    # Cities & splits
    # -------------------------
    a = _as_str(_get(store, "xfer.city_a", ""))
    b = _as_str(_get(store, "xfer.city_b", ""))
    splits = _as_list(_get(store, "xfer.splits", ("val", "test")))
    cals = _as_list(
        _get(store, "xfer.calib_modes", ("none", "source", "target"))
    )
    bsz = _get(store, "xfer.batch_size", None)

    if (not a) and (not b):
        out["cities"] = _warn("Set")
    elif (not a) or (not b):
        out["cities"] = _warn("A/B?")
    elif len(splits) == 0:
        out["cities"] = _warn("Split")
    elif len(cals) == 0:
        out["cities"] = _warn("Cal?")
    else:
        try:
            if bsz is not None and int(bsz) <= 0:
                out["cities"] = _warn("Fix")
            else:
                out["cities"] = _ok("OK")
        except Exception:
            out["cities"] = _warn("Fix")

    # -------------------------
    # Outputs & alignments
    # -------------------------
    align = _as_str(_get(store, "xfer.align_policy", "align_by_name_pad"))
    dyn = _get(store, "xfer.allow_reorder_dynamic", None)
    fut = _get(store, "xfer.allow_reorder_future", None)
    cov = _get(store, "xfer.interval_target", 0.80)
    ep = _as_str(_get(store, "xfer.load_endpoint", "serve"))

    wj = _get(store, "xfer.write_json", True)
    wc = _get(store, "xfer.write_csv", True)
    q = _get(store, "xfer.quantiles_override", None)

    ok = True
    tag = "OK"

    if align not in {"align_by_name_pad", "strict"}:
        ok = False
        tag = "Fix"

    if not _valid_opt_bool(dyn) or not _valid_opt_bool(fut):
        ok = False
        tag = "Fix"

    try:
        cv = float(cov)
        if (cv < 0.10) or (cv > 0.99):
            ok = False
            tag = "Cov?"
    except Exception:
        ok = False
        tag = "Cov?"

    if ep not in {"serve", "export"}:
        ok = False
        tag = "EP?"

    if (not isinstance(wj, bool)) or (not isinstance(wc, bool)):
        ok = False
        tag = "Fix"

    if not _valid_quantiles(q):
        ok = False
        tag = "Q?"

    out["outputs"] = _ok("OK") if ok else _warn(tag)

    # -------------------------
    # Strategy & warm-start
    # -------------------------
    strats = _as_list(_get(store, "xfer.strategies", None))
    modes = _as_list(_get(store, "xfer.rescale_modes", None))

    if len(strats) == 0:
        out["strategy"] = _warn("Set")
    else:
        warm_on = "warm" in set(str(s).strip() for s in strats)
        if not warm_on:
            out["strategy"] = _ok("OK")
        else:
            ws = _get(store, "xfer.warm_samples", 20000)
            we = _get(store, "xfer.warm_epochs", 3)
            wl = _get(store, "xfer.warm_lr", 1e-4)
            try:
                bad = (int(ws) <= 0) or (int(we) <= 0) or (float(wl) <= 0.0)
            except Exception:
                bad = True
            out["strategy"] = _warn("Fix") if bad else _ok("OK")

    # -------------------------
    # Results & view
    # -------------------------
    root = _as_str(_get(store, "results_root", ""))
    last = _as_str(_get(store, "xfer.last_output", ""))

    kind = _as_str(_get(store, "xfer.view_kind", "calib_panel"))
    split = _as_str(_get(store, "xfer.view_split", "val"))

    if last:
        out["results"] = _ok("OK")
    elif root:
        out["results"] = _ok("OK")
    else:
        out["results"] = _warn("Set")

    if kind not in {"calib_panel", "summary_panel"}:
        out["results"] = _warn("Fix")
    if split not in {"val", "test"}:
        out["results"] = _warn("Fix")

    return out



# # geoprior/ui/xfer/run/status.py
# # -*- coding: utf-8 -*-

# from __future__ import annotations

# from typing import Any, Dict, Iterable, Optional

# from ....config.prior_schema import FieldKey
# from ....config.store import GeoConfigStore


# __all__ = ["compute_xfer_nav"]


# def _ok(txt: str = "OK") -> Dict[str, str]:
#     return {"status": "ok", "text": txt}


# def _warn(txt: str = "Fix") -> Dict[str, str]:
#     return {"status": "warn", "text": txt}


# def _err(txt: str = "Err") -> Dict[str, str]:
#     return {"status": "err", "text": txt}


# def _get_fk(
#     store: GeoConfigStore,
#     key: str,
#     default: Any,
# ) -> Any:
#     try:
#         return store.get_value(FieldKey(key), default=default)
#     except Exception:
#         return default


# def _get_ui(
#     store: GeoConfigStore,
#     key: str,
#     default: Any,
# ) -> Any:
#     try:
#         return store.get(key, default)
#     except Exception:
#         return default


# def _pick(
#     store: GeoConfigStore,
#     keys: Iterable[str],
#     default: Any = None,
# ) -> Any:
#     """
#     Return first non-empty value among candidate keys.
#     Tries UI store first, then FieldKey lookup.
#     """
#     for k in keys:
#         v = _get_ui(store, k, None)
#         if v not in (None, "", [], {}, ()):
#             return v
#     for k in keys:
#         v = _get_fk(store, k, None)
#         if v not in (None, "", [], {}, ()):
#             return v
#     return default


# def _str_ok(v: Any) -> bool:
#     return bool(str(v or "").strip())


# def _bool(v: Any) -> Optional[bool]:
#     if isinstance(v, bool):
#         return v
#     if v in (0, 1):
#         return bool(v)
#     if isinstance(v, str):
#         s = v.strip().lower()
#         if s in {"true", "yes", "on"}:
#             return True
#         if s in {"false", "no", "off"}:
#             return False
#     return None


# def _int(v: Any) -> Optional[int]:
#     try:
#         return int(v)
#     except Exception:
#         return None


# def compute_xfer_nav(
#     store: GeoConfigStore,
# ) -> Dict[str, Dict[str, str]]:
#     """
#     Compute navigator chip statuses for Xfer RUN.

#     Keys
#     ----
#     - cities
#     - outputs
#     - strategy
#     - results

#     Notes
#     -----
#     Store-driven only (no path existence checks).
#     """

#     out: Dict[str, Dict[str, str]] = {}

#     # ---------------- Cities & splits ----------------
#     city_a = _pick(
#         store,
#         [
#             "xfer.city_a",
#             "xfer.a.city",
#             "xfer.run.city_a",
#         ],
#         "",
#     )
#     city_b = _pick(
#         store,
#         [
#             "xfer.city_b",
#             "xfer.b.city",
#             "xfer.run.city_b",
#         ],
#         "",
#     )

#     a_ok = _str_ok(city_a)
#     b_ok = _str_ok(city_b)

#     tr = _bool(
#         _pick(
#             store,
#             [
#                 "xfer.split.train",
#                 "xfer.split_train",
#                 "xfer.train",
#             ],
#             False,
#         )
#     )
#     va = _bool(
#         _pick(
#             store,
#             [
#                 "xfer.split.val",
#                 "xfer.split_val",
#                 "xfer.val",
#             ],
#             False,
#         )
#     )
#     te = _bool(
#         _pick(
#             store,
#             [
#                 "xfer.split.test",
#                 "xfer.split_test",
#                 "xfer.test",
#             ],
#             False,
#         )
#     )

#     any_split = bool(tr) or bool(va) or bool(te)

#     if (a_ok and b_ok) and any_split:
#         out["cities"] = _ok("OK")
#     elif a_ok ^ b_ok:
#         out["cities"] = _warn("A/B")
#     else:
#         out["cities"] = _warn("Set")

#     if (a_ok and b_ok) and (not any_split):
#         out["cities"] = _warn("Split")

#     # -------------- Outputs & alignments -------------
#     root = _pick(
#         store,
#         [
#             "xfer.results_root",
#             "xfer.root",
#             "xfer.out_root",
#         ],
#         "",
#     )

#     batch = _int(
#         _pick(
#             store,
#             [
#                 "xfer.batch",
#                 "xfer.batch_size",
#                 "xfer.run.batch",
#             ],
#             0,
#         )
#     )

#     rescale = _bool(
#         _pick(
#             store,
#             [
#                 "xfer.rescale",
#                 "xfer.do_rescale",
#                 "xfer.align.rescale",
#             ],
#             False,
#         )
#     )

#     if not _str_ok(root):
#         out["outputs"] = _warn("Set")
#     elif (batch is None) or (batch <= 0):
#         out["outputs"] = _warn("Fix")
#     elif rescale is None:
#         out["outputs"] = _warn("Fix")
#     else:
#         out["outputs"] = _ok("OK")

#     # -------------- Strategy & warm-start ------------
#     warm = _bool(
#         _pick(
#             store,
#             [
#                 "xfer.warm_start",
#                 "xfer.strategy.warm_start",
#                 "xfer.warmstart",
#             ],
#             False,
#         )
#     )

#     # Calibration choice (None / Source / Target)
#     cal_mode = _pick(
#         store,
#         [
#             "xfer.calibration",
#             "xfer.calibration_mode",
#             "xfer.calib",
#         ],
#         "",
#     )
#     cal_mode_s = str(cal_mode or "").strip().lower()

#     cal_none = _bool(
#         _pick(
#             store,
#             [
#                 "xfer.cal.none",
#                 "xfer.cal_none",
#             ],
#             False,
#         )
#     )
#     cal_src = _bool(
#         _pick(
#             store,
#             [
#                 "xfer.cal.source",
#                 "xfer.cal_source",
#             ],
#             False,
#         )
#     )
#     cal_tgt = _bool(
#         _pick(
#             store,
#             [
#                 "xfer.cal.target",
#                 "xfer.cal_target",
#             ],
#             False,
#         )
#     )

#     picked = 0
#     for v in (cal_none, cal_src, cal_tgt):
#         if bool(v):
#             picked += 1

#     if cal_mode_s in {"none", "source", "target"}:
#         # ok
#         pass
#     elif picked == 1:
#         # ok
#         pass
#     elif picked > 1:
#         out["strategy"] = _warn("Fix")
#     else:
#         out["strategy"] = _warn("Set")

#     if "strategy" not in out:
#         if warm is None:
#             out["strategy"] = _warn("Fix")
#         elif bool(warm):
#             out["strategy"] = _ok("OK")
#         else:
#             out["strategy"] = _ok("OK")

#     # ---------------- Results & view -----------------
#     last_out = _pick(
#         store,
#         [
#             "xfer.last_output",
#             "xfer.last_out",
#             "xfer.output_dir",
#         ],
#         "",
#     )

#     view_kind = _pick(
#         store,
#         [
#             "xfer.view.kind",
#             "xfer.view_kind",
#             "xfer.viewer.kind",
#         ],
#         "",
#     )

#     if not _str_ok(last_out):
#         out["results"] = _warn("Run")
#     elif not _str_ok(view_kind):
#         out["results"] = _warn("View")
#     else:
#         out["results"] = _ok("OK")

#     return out
