# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>

"""
Backend helpers for configuring the TensorFlow runtime (devices, threads,
and GPU memory policy) in a single place.

This module delegates TensorFlow import and logging configuration to
:mod:`fusionlab.compat.tf` and exposes a small, stable API:

- summarize_tf_devices
- configure_tf
- configure_tf_from_cfg
"""

from __future__ import annotations

import os
import warnings
from typing import Any, Dict, Mapping, Optional, Callable

try:
    # Centralized TF import / HAS_TF flag
    from ..compat import tf as _tf_compat
except Exception:  # pragma: no cover - very defensive
    _tf_compat = None

if _tf_compat is not None:
    HAS_TF: bool = bool(getattr(_tf_compat, "HAS_TF", False))
    tf = getattr(_tf_compat, "tf", None)
else:
    HAS_TF = False
    tf = None  # type: ignore[assignment]

TF_MESSAGE = (
    "TensorFlow is not available. Device and threading configuration "
    "are disabled. Install TensorFlow to enable GPU/CPU control."
)

LogFn = Optional[Callable[[str], None]]


def _get_logger(logger: LogFn) -> Callable[[str], None]:
    """
    Normalize logger to a callable; fall back to print().
    """
    if logger is None:
        return print

    def _wrapped(msg: str) -> None:
        try:
            logger(msg)
        except Exception:
            print(msg)

    return _wrapped


def summarize_tf_devices() -> Dict[str, Any]:
    """
    Lightweight summary of available TensorFlow physical devices.

    Returns
    -------
    summary : dict
        Contains keys::

            has_tf : bool
            cpus   : list
            gpus   : list
            num_cpus : int
            num_gpus : int
    """
    if not HAS_TF or tf is None:
        return {
            "has_tf": False,
            "cpus": [],
            "gpus": [],
            "num_cpus": 0,
            "num_gpus": 0,
        }

    cpus = tf.config.list_physical_devices("CPU")
    gpus = tf.config.list_physical_devices("GPU")
    return {
        "has_tf": True,
        "cpus": cpus,
        "gpus": gpus,
        "num_cpus": len(cpus),
        "num_gpus": len(gpus),
    }


def configure_tf(
    device: str = "auto",           # "auto" | "cpu" | "gpu"
    num_intra_threads: Optional[int] = None,
    num_inter_threads: Optional[int] = None,
    gpu_memory_growth: bool = True,
    gpu_memory_limit_mb: Optional[int] = None,
    logger: LogFn = None,           # if None, use print()
) -> Dict[str, Any]:
    """
    Configure TensorFlow runtime (device visibility, CPU threading and
    GPU memory policy).

    Parameters
    ----------
    device : {"auto", "cpu", "gpu"}, default="auto"
        - "auto": use GPU if available, else CPU.
        - "cpu" : hide all GPUs and run on CPU only.
        - "gpu" : expect at least one GPU; if none is found, we fall
          back to CPU but keep the requested mode in the returned
          summary.

    num_intra_threads, num_inter_threads : int or None, default=None
        Desired intra-op / inter-op thread counts. Values <= 0 are
        treated as ``None`` (leave TensorFlow defaults untouched).

    gpu_memory_growth : bool, default=True
        If ``True`` and no explicit limit is requested, enable "allow
        growth" for all visible GPUs.

    gpu_memory_limit_mb : int or None, default=None
        Optional per-process GPU memory cap (in megabytes). When set
        to a positive value, this takes precedence over
        ``gpu_memory_growth`` and creates logical devices with the
        requested memory limit. Values <= 0 are treated as ``None``.

    logger : callable or None, default=None
        Optional logging callback. If ``None``, falls back to ``print``.

    Returns
    -------
    info : dict
        A small summary of the effective configuration, including::

            has_tf
            device_mode_requested
            device_mode_effective
            num_cpus
            num_gpus
            visible_gpus
            intra_threads
            inter_threads
            gpu_memory_growth
            gpu_memory_limit_mb
    """
    log = _get_logger(logger)
    info: Dict[str, Any] = {
        "has_tf": HAS_TF and tf is not None,
        "device_mode_requested": device,
        "device_mode_effective": None,
        "num_cpus": 0,
        "num_gpus": 0,
        "visible_gpus": [],
        "intra_threads": None,
        "inter_threads": None,
        "gpu_memory_growth": None,
        "gpu_memory_limit_mb": None,
    }

    if not HAS_TF or tf is None:
        log(f"[TF] {TF_MESSAGE}")
        info["device_mode_effective"] = "none"
        return info

    # Normalise device string
    mode = (device or "auto").strip().lower()
    if mode not in {"auto", "cpu", "gpu"}:
        warnings.warn(
            f"Unknown device mode {mode!r}; falling back to 'auto'."
        )
        mode = "auto"

    # Discover physical devices
    cpus = tf.config.list_physical_devices("CPU")
    gpus = tf.config.list_physical_devices("GPU")
    info["num_cpus"] = len(cpus)
    info["num_gpus"] = len(gpus)

    # Decide effective mode
    eff_mode = mode
    if mode == "gpu" and not gpus:
        log("[TF] Requested GPU mode but no GPUs found; using CPU.")
        eff_mode = "cpu"
    if mode == "auto" and not gpus:
        eff_mode = "cpu"

    visible_gpus = []  # type: ignore[var-annotated]

    # --- GPU visibility & memory policy --------------------------------
    try:
        if eff_mode == "cpu":
            # Hide all GPUs (if any)
            try:
                tf.config.set_visible_devices([], "GPU")
                log("[TF] Configured to run on CPU only (GPUs hidden).")
            except RuntimeError as exc:
                # Devices already initialised; nothing we can do
                log(f"[TF] Could not hide GPUs (already initialised): {exc}")
            visible_gpus = []
        else:
            # Leave GPUs visible; apply memory policy if GPUs exist
            visible_gpus = list(gpus)
            mem_limit = (
                int(gpu_memory_limit_mb) if gpu_memory_limit_mb else None
            )

            if visible_gpus and mem_limit is not None and mem_limit > 0:
                # Memory limit takes precedence over growth
                try:
                    logical_cfg = [
                        tf.config.LogicalDeviceConfiguration(
                            memory_limit=mem_limit
                        )
                    ]
                    for gpu in visible_gpus:
                        tf.config.set_logical_device_configuration(
                            gpu, logical_cfg
                        )
                    info["gpu_memory_limit_mb"] = mem_limit
                    info["gpu_memory_growth"] = False
                    log(
                        "[TF] Set GPU memory limit to "
                        f"{mem_limit} MB for each visible GPU."
                    )
                except RuntimeError as exc:
                    log(
                        "[TF] Could not set GPU memory limit "
                        f"(devices already initialised?): {exc}"
                    )
            elif visible_gpus and gpu_memory_growth:
                try:
                    for gpu in visible_gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                    info["gpu_memory_growth"] = True
                    log("[TF] Enabled GPU memory growth.")
                except RuntimeError as exc:
                    log(
                        "[TF] Could not enable GPU memory growth "
                        f"(devices already initialised?): {exc}"
                    )
    except Exception as exc:  # pragma: no cover - safety net
        log(f"[TF] Unexpected error while configuring GPUs: {exc}")

    info["visible_gpus"] = [getattr(d, "name", str(d)) for d in visible_gpus]
    info["device_mode_effective"] = eff_mode

    # --- CPU threading --------------------------------------------------
    def _norm_threads(val: Optional[int]) -> Optional[int]:
        if val is None:
            return None
        try:
            iv = int(val)
        except Exception:
            return None
        return iv if iv > 0 else None

    intra = _norm_threads(num_intra_threads)
    inter = _norm_threads(num_inter_threads)

    if intra is not None:
        try:
            tf.config.threading.set_intra_op_parallelism_threads(intra)
            info["intra_threads"] = intra
            # Keep BLAS/OpenMP in sync where possible
            os.environ["OMP_NUM_THREADS"] = str(intra)
            os.environ["MKL_NUM_THREADS"] = str(intra)
            log(f"[TF] Set intra-op threads to {intra}.")
        except Exception as exc:  # pragma: no cover - env specific
            log(f"[TF] Could not set intra-op threads: {exc}")

    if inter is not None:
        try:
            tf.config.threading.set_inter_op_parallelism_threads(inter)
            info["inter_threads"] = inter
            log(f"[TF] Set inter-op threads to {inter}.")
        except Exception as exc:  # pragma: no cover - env specific
            log(f"[TF] Could not set inter-op threads: {exc}")

    return info


def configure_tf_from_cfg(
    cfg: Mapping[str, Any],
    logger: LogFn = None,
) -> Dict[str, Any]:
    """
    Read TF device options from a flat config dict and call
    :func:`configure_tf`.

    Expected keys in ``cfg`` (all optional)::

        TF_DEVICE_MODE          : "auto" | "cpu" | "gpu"
        TF_INTRA_THREADS        : int or None
        TF_INTER_THREADS        : int or None
        TF_GPU_ALLOW_GROWTH     : bool
        TF_GPU_MEMORY_LIMIT_MB  : int or None

    Returns
    -------
    info : dict
        Whatever :func:`configure_tf` returns.
    """

    def _as_int_or_none(key: str) -> Optional[int]:
        val = cfg.get(key, None)
        if val is None:
            return None
        try:
            iv = int(val)
        except Exception:
            return None
        return iv if iv > 0 else None

    dev_mode = str(cfg.get("TF_DEVICE_MODE", "auto")).strip().lower()
    intra = _as_int_or_none("TF_INTRA_THREADS")
    inter = _as_int_or_none("TF_INTER_THREADS")

    allow_growth_raw = cfg.get("TF_GPU_ALLOW_GROWTH", True)
    allow_growth = bool(allow_growth_raw)

    mem_raw = cfg.get("TF_GPU_MEMORY_LIMIT_MB", None)
    mem_limit: Optional[int]
    try:
        mem_limit = int(mem_raw) if mem_raw is not None else None
    except Exception:
        mem_limit = None
    if mem_limit is not None and mem_limit <= 0:
        mem_limit = None

    info = configure_tf(
        device=dev_mode or "auto",
        num_intra_threads=intra,
        num_inter_threads=inter,
        gpu_memory_growth=allow_growth,
        gpu_memory_limit_mb=mem_limit,
        logger=logger,
    )

    log = _get_logger(logger)
    log(
        "[TF] Device configured from cfg: "
        f"mode={dev_mode or 'auto'}, "
        f"intra={intra}, inter={inter}, "
        f"allow_growth={allow_growth}, "
        f"mem_limit_mb={mem_limit}."
    )

    return info


__all__ = [
    "HAS_TF",
    "TF_MESSAGE",
    "summarize_tf_devices",
    "configure_tf",
    "configure_tf_from_cfg",
]
