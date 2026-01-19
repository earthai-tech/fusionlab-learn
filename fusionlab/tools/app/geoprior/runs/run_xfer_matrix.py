# geoprior/runs/run_xfer_matrix.py
# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>

"""geoprior.runs.run_xfer_matrix

GUI/backend entry-point for transferability evaluation.

This wraps the v3.2 xfer engine (runs/xfer/) and keeps
API compatibility with XferMatrixJob and backend calls.

Key idea
--------
- stage5.py was CLI-only.
- xfer_utils.py + xfer_core.py are the new engine.
- run_xfer_matrix() adapts GUI/store -> xfer_core.run_plan.
"""

from __future__ import annotations

import os
from typing import Any, Callable, Dict, Optional, Sequence


from ..config.store import GeoConfigStore

from .xfer.xfer_core import WarmStartConfig
from .xfer.xfer_core import build_plan
from .xfer.xfer_core import run_plan


LogFn = Callable[[str], None]
StopCheckFn = Callable[[], bool]
ProgressFn = Callable[[float, Optional[str]], None]


def _log_fn(logger: Optional[LogFn]) -> LogFn:
    return logger if callable(logger) else print


def _safe_float(x: Any, default: float) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _safe_int(x: Any, default: int) -> int:
    try:
        return int(x)
    except Exception:
        return int(default)


def _safe_bool(x: Any, default: bool) -> bool:
    try:
        return bool(x)
    except Exception:
        return bool(default)


def _safe_seq(
    x: Any,
    default: Sequence[Any],
) -> Sequence[Any]:
    if x is None:
        return default
    if isinstance(x, (list, tuple)):
        return x
    return default


def _maybe_remove(path: Optional[str]) -> None:
    if not path:
        return
    try:
        if os.path.isfile(path):
            os.remove(path)
    except Exception:
        return


def run_xfer_matrix(
    city_a: str,
    city_b: str,
    *,
    store: Optional[GeoConfigStore] = None,
    results_dir: Optional[str] = None,
    results_root: Optional[str] = None,
    splits: Sequence[str] = ("val", "test"),
    calib_modes: Sequence[str] = (
        "none",
        "source",
        "target",
    ),
    rescale_to_source: bool = False,
    rescale_modes: Optional[Sequence[str]] = None,
    strategies: Optional[Sequence[str]] = None,
    batch_size: int = 32,
    quantiles_override: Optional[Sequence[float]] = None,
    out_dir: Optional[str] = None,
    write_json: bool = True,
    write_csv: bool = True,
    logger: Optional[LogFn] = None,
    stop_check: Optional[StopCheckFn] = None,
    progress_callback: Optional[ProgressFn] = None,
    model_name: str = "GeoPriorSubsNet",
    prefer_tuned: bool = True,
    align_policy: str = "align_by_name_pad",
    allow_reorder_dynamic: Optional[bool] = None,
    allow_reorder_future: Optional[bool] = None,
    warm_split: Optional[str] = None,
    warm_samples: Optional[int] = None,
    warm_frac: Optional[float] = None,
    warm_epochs: Optional[int] = None,
    warm_lr: Optional[float] = None,
    warm_seed: Optional[int] = None,
    **kws: Any,
) -> Dict[str, Any]:
    """Run transfer evaluation for (city_a, city_b).

    Parameters mirror the legacy GUI runner.

    Notes
    -----
    - `model_name` is accepted for backward compat.
      v3.2 xfer_core currently targets GeoPriorSubsNet.
    - `results_dir` and `results_root` are aliases.
    """

    log = _log_fn(logger)

    def _progress(v: float, msg: Optional[str]) -> None:
        if progress_callback is None:
            return
        try:
            vv = max(0.0, min(1.0, float(v)))
        except Exception:
            vv = 0.0
        try:
            progress_callback(vv, msg)
        except Exception:
            return

    # -------------------------------------------------
    # Store-driven defaults (optional)
    # -------------------------------------------------
    try:
        if store is not None:
            results_dir = (
                results_dir
                or results_root
                or store.get("results_root", None)
                or store.get("results_dir", None)
            )

            splits = tuple(
                _safe_seq(
                    store.get("xfer.splits", None),
                    splits,
                )
            )

            calib_modes = tuple(
                _safe_seq(
                    store.get("xfer.calib_modes", None),
                    calib_modes,
                )
            )

            strategies = tuple(
                _safe_seq(
                    store.get("xfer.strategies", None),
                    strategies or ("baseline", "xfer"),
                )
            )

            batch_size = _safe_int(
                store.get("xfer.batch_size", batch_size),
                batch_size,
            )

            prefer_tuned = _safe_bool(
                store.get("xfer.prefer_tuned", prefer_tuned),
                prefer_tuned,
            )

            align_policy = str(
                store.get("xfer.align_policy", align_policy)
            )

            if rescale_modes is None:
                rescale_modes = store.get(
                    "xfer.rescale_modes",
                    None,
                )

            rescale_to_source = _safe_bool(
                store.get(
                    "xfer.rescale_to_source",
                    rescale_to_source,
                ),
                rescale_to_source,
            )

            warm_split = store.get(
                "xfer.warm_split",
                warm_split,
            )
            warm_samples = store.get(
                "xfer.warm_samples",
                warm_samples,
            )
            warm_frac = store.get(
                "xfer.warm_frac",
                warm_frac,
            )
            warm_epochs = store.get(
                "xfer.warm_epochs",
                warm_epochs,
            )
            warm_lr = store.get("xfer.warm_lr", warm_lr)
            warm_seed = store.get(
                "xfer.warm_seed",
                warm_seed,
            )
    except Exception:
        # Never fail because store types are odd.
        pass

    # -------------------------------------------------
    # Normalize / derive options
    # -------------------------------------------------
    results_dir = results_dir or results_root or "results"

    if strategies is None:
        strategies = ("baseline", "xfer")

    if rescale_modes is None:
        if rescale_to_source:
            rescale_modes = ("strict",)
        else:
            rescale_modes = ("as_is",)

    ap = (align_policy or "").strip().lower()
    if allow_reorder_dynamic is None:
        allow_reorder_dynamic = ap != "strict"
    if allow_reorder_future is None:
        allow_reorder_future = ap != "strict"

    warm = WarmStartConfig()
    if warm_split is not None:
        warm.split = str(warm_split)
    if warm_samples is not None:
        warm.samples = _safe_int(warm_samples, warm.samples)
    if warm_frac is not None:
        warm.frac = _safe_float(warm_frac, 0.0)
    if warm_epochs is not None:
        warm.epochs = _safe_int(warm_epochs, warm.epochs)
    if warm_lr is not None:
        warm.lr = _safe_float(warm_lr, warm.lr)
    if warm_seed is not None:
        warm.seed = _safe_int(warm_seed, warm.seed)

    # -------------------------------------------------
    # Build plan + run
    # -------------------------------------------------
    _progress(0.01, "XFER: building plan")

    plan = build_plan(
        city_a=city_a,
        city_b=city_b,
        results_dir=str(results_dir),
        splits=tuple(splits),
        strategies=tuple(strategies),
        calib_modes=tuple(calib_modes),
        rescale_modes=tuple(rescale_modes),
        batch_size=int(batch_size),
        quantiles=list(quantiles_override)
        if quantiles_override
        else None,
        allow_reorder_dynamic=bool(allow_reorder_dynamic),
        allow_reorder_future=bool(allow_reorder_future),
        prefer_tuned=bool(prefer_tuned),
        out_dir=out_dir,
        warm=warm,
    )

    if model_name and model_name != "GeoPriorSubsNet":
        log(
            "[xfer] NOTE: model_name is ignored in v3.2 "
            f"(got {model_name!r})."
        )

    def _prog_i(i: int, n: int) -> None:
        msg = f"XFER: {i}/{n}"
        _progress(i / max(1, n), msg)

    try:
        out = run_plan(
            plan=plan,
            log_fn=logger,
            progress_fn=_prog_i,
            stop_check=stop_check,
        )
    except Exception as e:
        log(f"[xfer] ERROR: {e}")
        return {
            "out_dir": out_dir,
            "results": [],
            "json_path": None,
            "csv_path": None,
            "error": str(e),
        }

    json_path = out.get("json_path")
    csv_path = out.get("csv_path")

    if not write_json:
        _maybe_remove(json_path)
        json_path = None

    if not write_csv:
        _maybe_remove(csv_path)
        csv_path = None

    result: Dict[str, Any] = {
        "out_dir": out.get("out_dir"),
        "results": out.get("results", []),
        "json_path": json_path,
        "csv_path": csv_path,
    }

    # -------------------------------------------------
    # Store last-run pointers (best-effort)
    # -------------------------------------------------
    try:
        if store is not None:
            store.set("xfer.last_out_dir", result["out_dir"])
            store.set("xfer.last_json", json_path)
            store.set("xfer.last_csv", csv_path)
            store.set(
                "xfer.last_n",
                len(result.get("results") or []),
            )
    except Exception:
        pass

    _progress(1.0, "XFER: done")
    return result
