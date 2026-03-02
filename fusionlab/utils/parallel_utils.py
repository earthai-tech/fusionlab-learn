# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio

from __future__ import annotations

import os
import subprocess
from typing import Dict, List, Optional

def resolve_n_jobs(n_jobs: int) -> int:
    try:
        n = int(n_jobs)
    except Exception:
        return 1

    if n == 0 or n == 1:
        return 1
    if n < 0:
        return int(os.cpu_count() or 1)
    return max(1, n)


def threads_per_job(
    *,
    n_jobs: int,
    threads: int = 0,
    reserve: int = 1,
) -> int:
    if int(threads) > 0:
        return int(threads)

    cpu = int(os.cpu_count() or 1)
    cpu = max(1, cpu - int(reserve))
    return max(1, cpu // max(1, int(n_jobs)))


def apply_thread_env(
    env: Dict[str, str],
    *,
    n_jobs: int,
    threads: int = 0,
    reserve: int = 1,
) -> Dict[str, str]:
    out = dict(env)
    t = threads_per_job(
        n_jobs=n_jobs,
        threads=threads,
        reserve=reserve,
    )

    inter = max(1, min(4, t // 2))

    out["OMP_NUM_THREADS"] = str(t)
    out["MKL_NUM_THREADS"] = str(t)
    out["NUMEXPR_NUM_THREADS"] = str(t)

    out["TF_NUM_INTRAOP_THREADS"] = str(t)
    out["TF_NUM_INTEROP_THREADS"] = str(inter)
    return out


def apply_tf_threading(
    *,
    intra: int,
    inter: int,
) -> None:
    import tensorflow as tf

    tf.config.threading.set_intra_op_parallelism_threads(
        int(intra)
    )
    tf.config.threading.set_inter_op_parallelism_threads(
        int(inter)
    )
    

def _split_ids(s: str) -> List[str]:
    s = str(s).strip()
    if not s:
        return []
    out: List[str] = []
    for part in s.split(","):
        p = part.strip()
        if p:
            out.append(p)
    return out


def detect_gpu_ids(
    *,
    env: Optional[Dict[str, str]] = None,
) -> List[str]:
    e = env or os.environ

    # Respect an existing restriction first
    cvd = e.get("CUDA_VISIBLE_DEVICES", None)
    if cvd is not None:
        ids = _split_ids(cvd)
        return ids

    # Try nvidia-smi (fast, no TF import)
    try:
        r = subprocess.run(
            ["nvidia-smi", "-L"],
            capture_output=True,
            text=True,
            check=False,
        )
        if r.returncode == 0 and r.stdout.strip():
            lines = r.stdout.splitlines()
            return [str(i) for i in range(len(lines))]
    except Exception:
        pass

    # Fall back to TF if available
    try:
        import tensorflow as tf

        g = tf.config.list_physical_devices("GPU")
        return [str(i) for i in range(len(g))]
    except Exception:
        return []


def resolve_device(
    device: str,
    *,
    env: Optional[Dict[str, str]] = None,
) -> str:
    d = str(device).strip().lower()
    if d in {"cpu"}:
        return "cpu"
    if d in {"gpu"}:
        return "gpu"
    # auto
    return "gpu" if detect_gpu_ids(env=env) else "cpu"


def resolve_gpu_ids(
    gpu_ids: Optional[List[str]],
    *,
    env: Optional[Dict[str, str]] = None,
) -> List[str]:
    if gpu_ids:
        return [str(x).strip() for x in gpu_ids if str(x).strip()]
    return detect_gpu_ids(env=env)


def pick_gpu_id(
    idx: int,
    gpu_ids: List[str],
) -> Optional[str]:
    if not gpu_ids:
        return None
    return gpu_ids[int(idx) % len(gpu_ids)]


def apply_gpu_env(
    env: Dict[str, str],
    *,
    gpu_id: Optional[str],
    allow_growth: bool = True,
) -> Dict[str, str]:
    out = dict(env)
    if gpu_id is None:
        out["CUDA_VISIBLE_DEVICES"] = ""
        return out

    out["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    out["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    if allow_growth:
        out["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

    return out