# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>
"""
run_sensitivity.py

Driver to run a (lambda_cons, lambda_prior) sensitivity grid
for GeoPriorSubsNet using the existing Stage-2 training script.

This script calls sensitivity.py multiple times with environment
overrides. Each run should write its own ablation record
(entry in ablation_records/ablation_record.jsonl), which your
make_supp_figS6_ablations.py later aggregates.

Core overrides (expected by sensitivity.py)
--------------------------------------------
- EPOCHS_OVERRIDE
- PDE_MODE_OVERRIDE
- LAMBDA_CONS_OVERRIDE
- LAMBDA_PRIOR_OVERRIDE

Optional "deconfounding" overrides (safe to export even if
sensivity.py ignores some of them; you can wire them later)
-----------------------------------------------------------
- TRAINING_STRATEGY_OVERRIDE
- Q_POLICY_OVERRIDE
- SUBS_RESID_POLICY_OVERRIDE
- ALLOW_SUBS_RESIDUAL_OVERRIDE
- LAMBDA_Q_OVERRIDE
- PHYSICS_WARMUP_STEPS_OVERRIDE
- PHYSICS_RAMP_STEPS_OVERRIDE
- LAMBDA_GW_OVERRIDE
- LAMBDA_SMOOTH_OVERRIDE
- LAMBDA_BOUNDS_OVERRIDE
- LAMBDA_MV_OVERRIDE

Driver to run a (lambda_cons, lambda_prior) sensitivity grid
for GeoPriorSubsNet using the Stage-2 sensitivity script.

Resume mechanism
----------------
On restart, the script scans existing ablation_record.jsonl files
under the results directory and skips runs that already finished.

A run is considered "done" if an ablation record exists containing:
  - pde_mode
  - lambda_cons
  - lambda_prior
(and matching CITY when available).

Usage
------
set CITY=zhongshan
python nat.com/run_lambda_sensitivity.py --epochs 20


to force rerun everything:

python nat.com/run_sensitivity.py --epochs 20 --no-resume

results live elsewhere: 

python nat.com/run_sensitivity.py --epochs 20 \
  --scan-root F:/repositories/fusionlab-learn/results/zhongshan
  
python nat.com/run_sensitivity.py --epochs 20 --inprocess --fast

python nat.com/run_sensitivity.py --epochs 20 --gold --eval-max-batches 50 --fast

python nat.com/run_sensitivity.py --epochs 10 --fast --n-jobs -1   

python nat.com/run_sensitivity.py --gold --epochs 10 --fast --threads 20

"""

from __future__ import annotations

import argparse
import itertools
import json
import runpy
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from concurrent.futures import (
    ThreadPoolExecutor,
    as_completed,
)
    
from fusionlab.utils import (
    default_results_dir, 
    resolve_n_jobs,
    threads_per_job,
    apply_tf_threading,
    apply_thread_env,
    resolve_device,
    resolve_gpu_ids,
    pick_gpu_id,
    apply_gpu_env,
)
        
from sensitivity_lib import (
    build_context,
    run_one as run_one_gold,
    cleanup_between_runs,
)
    
TRAIN_SCRIPT_DEFAULT = Path(__file__).with_name(
    "sensitivity.py"
)
# 0 0.05 0.2 1.0
DEFAULT_LCONS: List[float] = [
    0.0,
    0.01,
    0.05,
    0.1,
    0.2,
    0.5,
    1.0,
]

DEFAULT_LPRIOR: List[float] = [
    0.0,
    0.01,
    0.05,
    0.1,
    0.2,
    0.5,
    1.0,
]

DEFAULT_PDE_MODES: List[str] = ["both"] # ["none", "both" ]


def _fmt_float(x: float) -> str:
    # Stable-ish string key for floats in configs.
    # Uses "g" to match your tag style.
    try:
        return f"{float(x):g}"
    except Exception:
        return str(x)


def _norm_mode(x: str) -> str:
    return str(x).strip().lower()

def _canon_pde_mode(x: str) -> str:
    m = str(x).strip().lower()
    if m in {"both", "on", "true"}:
        return "on"
    if m in {"none", "off", "false"}:
        return "none"
    return m


@dataclass(frozen=True)
class RunSpec:
    pde_mode: str
    lambda_cons: float
    lambda_prior: float

    def key(self) -> str:
        pde = _canon_pde_mode(self.pde_mode)
        lc = _fmt_float(self.lambda_cons)
        lp = _fmt_float(self.lambda_prior)
        return f"pde={pde}|lcons={lc}|lprior={lp}"

    def tag(self) -> str:
        # Human-readable
        pde = str(self.pde_mode)
        lc = _fmt_float(self.lambda_cons)
        lp = _fmt_float(self.lambda_prior)
        return f"pde={pde}, lcons={lc}, lprior={lp}"

    def run_tag(self) -> str:
        # Filesystem-friendly (short)
        pde = _norm_mode(self.pde_mode)
        lc = _fmt_float(self.lambda_cons).replace(".", "p")
        lp = _fmt_float(self.lambda_prior).replace(".", "p")
        return f"pde_{pde}__lc_{lc}__lp_{lp}"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Run a lambda_cons / lambda_prior sensitivity "
            "grid using the stage2 sensitivity script."
        )
    )

    p.add_argument(
        "--train-script",
        type=str,
        default=str(TRAIN_SCRIPT_DEFAULT),
        help="Path to training script (sensitivity.py).",
    )

    p.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Epochs per run (short sensitivity runs).",
    )

    p.add_argument(
        "--pde-modes",
        type=str,
        nargs="+",
        default=DEFAULT_PDE_MODES,
        help="PDE modes to sweep (e.g. none both).",
    )

    p.add_argument(
        "--lcons",
        type=float,
        nargs="+",
        default=DEFAULT_LCONS,
        help="Grid for lambda_cons.",
    )

    p.add_argument(
        "--lprior",
        type=float,
        nargs="+",
        default=DEFAULT_LPRIOR,
        help="Grid for lambda_prior.",
    )
    p.add_argument(
        "--no-early-stopping",
        action="store_true",
        help=(
            "Disable EarlyStopping in "
            "sensitivity.py."
        ),
    )
    
    p.add_argument(
        "--fast",
        action="store_true",
        help=(
            "Skip plotting + calibration "
            "extras in sensitivity.py."
        ),
    )
    p.add_argument(
        "--eval-max-batches",
        type=int,
        default=None,
        help="Limit eval/export batches per run (speeds grids).",
    )

    p.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override BATCH_SIZE for stage-2 runs.",
    )

    p.add_argument(
        "--inprocess",
        action="store_true",
        help="Run sensitivity.py in-process (no subprocess).",
    )
    p.add_argument(
        "--gold",
        action="store_true",
        help=(
            "Gold mode: run sensitivity in-process via sensitivity_lib "
            "(reuses NPZ + tf.data pipelines). Fastest for grids."
        ),
    )
    p.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Parallel grid runs; -1=all CPUs.",
    )
    p.add_argument(
        "--threads",
        type=int,
        default=0,
        help="Threads per run (0=auto).",
    )

    p.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "gpu"],
        help="Device policy for runs.",
    )
    
    p.add_argument(
        "--gpu-ids",
        type=str,
        nargs="*",
        default=None,
        help=(
            "Explicit GPU ids, e.g. "
            "--gpu-ids 0 1"
        ),
    )
    
    p.add_argument(
        "--gpu-allow-growth",
        action="store_true",
        help="Enable TF GPU allow-growth.",
    )

    # -----------------------------
    # Optional deconfounding knobs
    # -----------------------------
    p.add_argument(
        "--strategy",
        type=str,
        default="data_first",
        choices=["data_first", "physics_first"],
        help="Training strategy override.",
    )

    p.add_argument(
        "--disable-q",
        action="store_true",
        help="Export overrides to force Q always off.",
    )

    p.add_argument(
        "--disable-subs-resid",
        action="store_true",
        help="Export overrides to disable subs residual.",
    )

    p.add_argument(
        "--no-physics-ramp",
        action="store_true",
        help="Set physics warmup/ramp steps to 0.",
    )

    p.add_argument(
        "--physics-warmup-steps",
        type=int,
        default=None,
        help="Override physics warmup steps.",
    )

    p.add_argument(
        "--physics-ramp-steps",
        type=int,
        default=None,
        help="Override physics ramp steps.",
    )

    p.add_argument(
        "--lambda-gw",
        type=float,
        default=None,
        help="Optional override for lambda_gw.",
    )
    p.add_argument(
        "--lambda-smooth",
        type=float,
        default=None,
        help="Optional override for lambda_smooth.",
    )
    p.add_argument(
        "--lambda-bounds",
        type=float,
        default=None,
        help="Optional override for lambda_bounds.",
    )
    p.add_argument(
        "--lambda-mv",
        type=float,
        default=None,
        help="Optional override for lambda_mv.",
    )
    p.add_argument(
        "--lambda-q",
        type=float,
        default=None,
        help="Optional override for lambda_q.",
    )

    # -----------------------------
    # Resume controls
    # -----------------------------
    p.add_argument(
        "--no-resume",
        action="store_true",
        help="Do NOT skip completed runs.",
    )

    p.add_argument(
        "--scan-root",
        type=str,
        default=None,
        help=(
            "Root directory to scan for prior "
            "ablation_record.jsonl files. "
            "Default: results_dir/CITY."
        ),
    )

    p.add_argument(
        "--state-file",
        type=str,
        default=None,
        help=(
            "Optional JSON state file to write progress "
            "(default: <scan_root>/lambda_sensitivity_state.json)."
        ),
    )

    # -----------------------------
    # Runner controls
    # -----------------------------
    p.add_argument(
        "--start",
        type=int,
        default=0,
        help="Start index in the remaining grid.",
    )

    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max number of runs (after --start).",
    )

    p.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle run order (deterministic with --seed).",
    )

    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for shuffling.",
    )

    p.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue grid even if a run fails.",
    )

    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned commands without executing.",
    )

    return p.parse_args()


def build_grid(
    pde_modes: Iterable[str],
    lcons: Iterable[float],
    lprior: Iterable[float],
) -> List[RunSpec]:
    out: List[RunSpec] = []
    for pde_mode in pde_modes:
        for lc, lp in itertools.product(lcons, lprior):
            out.append(
                RunSpec(
                    pde_mode=str(pde_mode),
                    lambda_cons=float(lc),
                    lambda_prior=float(lp),
                )
            )
    return out

def maybe_shuffle(
    runs: List[RunSpec],
    *,
    shuffle: bool,
    seed: int,
) -> List[RunSpec]:
    if not shuffle:
        return runs

    n = len(runs)
    if n <= 1:
        return runs

    idx = list(range(n))
    x = int(seed) & 0xFFFFFFFF
    for i in range(n - 1, 0, -1):
        x ^= (x << 13) & 0xFFFFFFFF
        x ^= (x >> 17) & 0xFFFFFFFF
        x ^= (x << 5) & 0xFFFFFFFF
        j = x % (i + 1)
        idx[i], idx[j] = idx[j], idx[i]

    return [runs[k] for k in idx]


def apply_runner_slicing(
    runs: List[RunSpec],
    *,
    start: int,
    limit: Optional[int],
) -> List[RunSpec]:
    if start < 0:
        start = 0
    out = runs[start:]
    if limit is None:
        return out
    if limit <= 0:
        return []
    return out[:limit]


def _default_scan_root(city: str) -> Path:
    # Prefer fusionlab's default_results_dir if available.
    # Fall back to ./results.
    try:
        
        root = Path(default_results_dir())
    except Exception:
        root = Path.cwd() / "results"

    if city and city != "<unknown>":
        return root / city
    return root


def _iter_ablation_jsonl_files(root: Path) -> Iterable[Path]:
    if not root.exists():
        return []
    # Typical layout: .../ablation_records/ablation_record.jsonl
    return root.rglob("ablation_record.jsonl")

def _load_completed_keys(
    scan_root: Path,
    *,
    city: str,
) -> Set[str]:
    done: Set[str] = set()
    for fp in _iter_ablation_jsonl_files(scan_root):
        try:
            with fp.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except Exception:
                        continue

                    # Filter by city when present
                    rec_city = rec.get("city", None)
                    if rec_city is not None:
                        if str(rec_city).lower() != str(city).lower():
                            continue

                    pde = _canon_pde_mode(rec.get("pde_mode"))
                    lc = rec.get("lambda_cons", None)
                    lp = rec.get("lambda_prior", None)
                    if pde is None or lc is None or lp is None:
                        continue

                    k = RunSpec(
                        pde_mode=str(pde),
                        lambda_cons=float(lc),
                        lambda_prior=float(lp),
                    ).key()
                    done.add(k)
        except Exception:
            continue
    return done

def _iter_done_json(scan_root: Path) -> Iterable[Path]:
    if not scan_root.exists():
        return []
    return scan_root.rglob("DONE.json")

def _load_completed_keys_from_done(
    scan_root: Path,
    *,
    city: str,
) -> Set[str]:
    done: Set[str] = set()

    for fp in _iter_done_json(scan_root):
        try:
            rec = json.loads(fp.read_text(encoding="utf-8"))
        except Exception:
            continue

        rec_city = rec.get("city", None)
        if rec_city is not None:
            if str(rec_city).lower() != str(city).lower():
                continue

        pde = rec.get("pde_mode", None)
        lc = rec.get("lambda_cons", None)
        lp = rec.get("lambda_prior", None)
        if pde is None or lc is None or lp is None:
            continue

        k = RunSpec(
            pde_mode=_canon_pde_mode(pde),
            lambda_cons=float(lc),
            lambda_prior=float(lp),
        ).key()
        done.add(k)

    return done

def _save_state(
    state_path: Path,
    *,
    city: str,
    scan_root: Path,
    completed_n: int,
    last_key: Optional[str],
) -> None:
    payload = {
        "city": city,
        "scan_root": str(scan_root),
        "completed_n": int(completed_n),
        "last_completed_key": last_key,
    }
    try:
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(
            json.dumps(payload, indent=2),
            encoding="utf-8",
        )
    except:
        # State is optional: never fail the run.
        return
    
def _worker_banner(
    *,
    mode: str,
    job_i: int,
    n_jobs: int,
    pool: int,
    run_tag: str,
    device: str,
    gpu_id: Optional[str],
) -> None:
    d = str(device).lower().strip()
    gid = "-" if gpu_id is None else str(gpu_id)
    prefix = f"[{mode}]"
    if d == "gpu":
        msg = (
            f"{prefix} job {job_i+1}/{n_jobs} | "
            f"pool={pool} | RUN_TAG={run_tag} | "
            f"GPU={gid}"
        )
    else:
        msg = (
            f"{prefix} job {job_i+1}/{n_jobs} | "
            f"pool={pool} | RUN_TAG={run_tag} | "
            f"CPU"
        )
    print(msg, flush=True)

def make_env(
    base_env: Dict[str, str],
    *,
    epochs: int,
    spec: RunSpec,
    strategy: str,
    disable_q: bool,
    disable_subs_resid: bool,
    no_physics_ramp: bool,
    physics_warmup_steps: Optional[int],
    physics_ramp_steps: Optional[int],
    lambda_gw: Optional[float],
    lambda_smooth: Optional[float],
    lambda_bounds: Optional[float],
    lambda_mv: Optional[float],
    lambda_q: Optional[float],
    no_early_stopping: bool,
    fast: bool,
    eval_max_batches: Optional[int],
    batch_size : Optional[int],
) -> Dict[str, str]:
    env = dict(base_env)

    # Core sweep
    env["PDE_MODE_OVERRIDE"] = str(spec.pde_mode)
    env["EPOCHS_OVERRIDE"] = str(int(epochs))
    env["LAMBDA_CONS_OVERRIDE"] = str(spec.lambda_cons)
    env["LAMBDA_PRIOR_OVERRIDE"] = str(spec.lambda_prior)
    
    env["SENS_WORKER_BANNER"] = "1"

    # Traceability
    env["RUN_TAG"] = spec.run_tag()
    env["DISABLE_EARLY_STOPPING"] = (
        "1" if no_early_stopping else "0"
    )
    env["FAST_SENSITIVITY"] = (
        "1" if fast else "0"
    )

    if eval_max_batches is not None:
        n = int(eval_max_batches)
        env["SENS_EVAL_MAX_BATCHES"] = str(n)

    if batch_size is not None:
        env["BATCH_SIZE_OVERRIDE"] = str(int(batch_size))
        
    # Optional controls
    env["TRAINING_STRATEGY_OVERRIDE"] = str(strategy)

    if disable_q:
        env["Q_POLICY_OVERRIDE"] = "always_off"
        env["LAMBDA_Q_OVERRIDE"] = "0.0"
    elif lambda_q is not None:
        env["LAMBDA_Q_OVERRIDE"] = str(lambda_q)

    if disable_subs_resid:
        env["SUBS_RESID_POLICY_OVERRIDE"] = "always_off"
        env["ALLOW_SUBS_RESIDUAL_OVERRIDE"] = "0"

    if no_physics_ramp:
        env["PHYSICS_WARMUP_STEPS_OVERRIDE"] = "0"
        env["PHYSICS_RAMP_STEPS_OVERRIDE"] = "0"
    else:
        if physics_warmup_steps is not None:
            env["PHYSICS_WARMUP_STEPS_OVERRIDE"] = str(
                int(physics_warmup_steps)
            )
        if physics_ramp_steps is not None:
            env["PHYSICS_RAMP_STEPS_OVERRIDE"] = str(
                int(physics_ramp_steps)
            )

    if lambda_gw is not None:
        env["LAMBDA_GW_OVERRIDE"] = str(lambda_gw)
    if lambda_smooth is not None:
        env["LAMBDA_SMOOTH_OVERRIDE"] = str(lambda_smooth)
    if lambda_bounds is not None:
        env["LAMBDA_BOUNDS_OVERRIDE"] = str(lambda_bounds)
    if lambda_mv is not None:
        env["LAMBDA_MV_OVERRIDE"] = str(lambda_mv)

    env["VERBOSE_OVERRIDE"] = "1"
    env["AUDIT_STAGES_OVERRIDE"] = "off"
    env["DEBUG_OVERRIDE"] = "0"
    env["LOG_Q_DIAGNOSTICS_OVERRIDE"] = "0"

    # env["Q_POLICY_OVERRIDE"] = "always_off"
    # env["LAMBDA_Q_OVERRIDE"] = "0.0"

    # env["SUBS_RESID_POLICY_OVERRIDE"] = "always_off"
    # env["ALLOW_SUBS_RESIDUAL_OVERRIDE"] = "0"

    # env["PHYSICS_WARMUP_STEPS_OVERRIDE"] = "0"
    # env["PHYSICS_RAMP_STEPS_OVERRIDE"] = "0"

    # env["LAMBDA_MV_OVERRIDE"] = "0.0"
    # env["MV_WEIGHT_OVERRIDE"] = "0.0"
    env["MV_WEIGHT_OVERRIDE"] = "0.0"
    if lambda_mv is None:
        env["LAMBDA_MV_OVERRIDE"] = "0.0"


    return env


def run_one_script(
    train_script: Path,
    *,
    env: Dict[str, str],
    dry_run: bool,
    inprocess: bool,
) -> None:
    cmd = [sys.executable, str(train_script)]
    if dry_run:
        print("[DryRun] " + " ".join(cmd))
        return

    if not inprocess:
        subprocess.run(cmd, env=env, check=True)
        return

    old_env = os.environ.copy()
    try:
        os.environ.update(env)
        runpy.run_path(str(train_script), run_name="__main__")
    finally:
        os.environ.clear()
        os.environ.update(old_env)
        try:
            import tensorflow as tf
            tf.keras.backend.clear_session()
        except Exception:
            pass
        import gc
        gc.collect()


def main() -> None:
    args = parse_args()

    train_script = Path(args.train_script)
    if not train_script.exists():
        raise SystemExit(
            "Cannot find training script at: "
            f"{train_script}"
        )

    base_env = os.environ.copy()
    city = base_env.get("CITY", "<unknown>")
    
    dev = resolve_device(args.device, env=base_env)
    gpus = []
    if dev == "gpu":
        gpus = resolve_gpu_ids(args.gpu_ids, env=base_env)
    
    if dev == "gpu" and not gpus:
        print("[Warn] device=gpu but no GPUs found.")
        print("       Falling back to CPU.")
        dev = "cpu"
    
    # Build full grid
    grid0 = build_grid(args.pde_modes, args.lcons, args.lprior)
    grid1 = maybe_shuffle(grid0, shuffle=args.shuffle, seed=args.seed)

    resume = not bool(args.no_resume)

    # Resolve scan root
    if args.scan_root is not None:
        scan_root = Path(args.scan_root)
    else:
        scan_root = _default_scan_root(city)

    # State file (optional)
    if args.state_file is not None:
        state_path = Path(args.state_file)
    else:
        state_path = scan_root / "lambda_sensitivity_state.json"

    completed: Set[str] = set()
    if resume:
        completed = _load_completed_keys_from_done(
            scan_root,
            city=city,
        )
        if not completed:
            # fallback: slower but robust
            completed = _load_completed_keys(
                scan_root,
                city=city,
            )

    # Filter completed BEFORE slicing
    if resume and completed:
        grid2: List[RunSpec] = []
        skipped = 0
        for spec in grid1:
            if spec.key() in completed:
                skipped += 1
                continue
            grid2.append(spec)
    else:
        grid2 = list(grid1)
        skipped = 0

    grid = apply_runner_slicing(
        grid2,
        start=args.start,
        limit=args.limit,
    )

    print("[Sensitivity] Setup")
    print(f"  CITY          : {city}")
    print(f"  train_script  : {train_script}")
    print(f"  epochs/run    : {args.epochs}")
    print(f"  pde_modes     : {list(args.pde_modes)}")
    print(f"  lcons grid    : {list(args.lcons)}")
    print(f"  lprior grid   : {list(args.lprior)}")
    print(f"  strategy      : {args.strategy}")
    print(f"  resume        : {resume}")
    print(f"  scan_root     : {scan_root}")
    print(f"  done_found    : {len(completed)}")
    print(f"  skipped_done  : {skipped}")
    print(f"  start         : {args.start}")
    print(f"  limit         : {args.limit}")
    print(f"  shuffle       : {bool(args.shuffle)}")
    print(f"  seed          : {args.seed}")
    print(f"  runs          : {len(grid)} / {len(grid0)}")
    print(f"  dry_run       : {bool(args.dry_run)}")
    print(
        "  continue_err  : "
        f"{bool(args.continue_on_error)}"
    )

    if not grid:
        print("[Sensitivity] No runs selected. Done.")
        _save_state(
            state_path,
            city=city,
            scan_root=scan_root,
            completed_n=len(completed),
            last_key=None,
        )
        return

    # ---------------------------------------------------------
    # GOLD MODE: cached context + in-process per-point runs
    # ---------------------------------------------------------
    if bool(args.gold):
        cpu = resolve_n_jobs(-1)
        t = threads_per_job(
            n_jobs=1,
            threads=int(args.threads or 0),
            reserve=1,
        )
        apply_tf_threading(intra=t, inter=max(1, min(4, t // 2)))
        
        if dev == "gpu":
            try:
                import tensorflow as tf
        
                for g in tf.config.list_physical_devices("GPU"):
                    tf.config.experimental.set_memory_growth(
                        g, True
                    )
            except:
                pass
    
        # Build cached context ONCE
        ctx = build_context(city=city, verbose=1)

        # If user didn’t pass scan_root, scan where gold runs live:
        # (inside Stage-1 run_dir)
        if args.scan_root is None:
            scan_root = Path(ctx.base_output_dir)
        else:
            scan_root = Path(args.scan_root)

        # Resume detection (reuse your existing DONE.json logic)
        resume = not bool(args.no_resume)
        completed: Set[str] = set()
        if resume:
            completed = _load_completed_keys_from_done(
                scan_root,
                city=city,
            )
            if not completed:
                completed = _load_completed_keys(
                    scan_root,
                    city=city,
                )

        # Filter completed BEFORE slicing (optional but recommended)
        if resume and completed:
            grid_gold: List[RunSpec] = []
            for spec in grid:
                if spec.key() in completed:
                    continue
                grid_gold.append(spec)
            grid = grid_gold

        failures: List[Tuple[int, str]] = []
        last_done: Optional[str] = None

        for i, spec in enumerate(grid):
            print("\n" + "=" * 62)
            print(f"[Sensitivity GOLD] Run {i+1}/{len(grid)}")
            print(f"  {spec.tag()}")
            print("=" * 62)

            # Build cfg overrides (direct cfg keys; no env needed)
            overrides: Dict[str, Any] = {
                "EPOCHS": int(args.epochs),
                "PDE_MODE_CONFIG": str(spec.pde_mode).strip().lower(),
                "LAMBDA_CONS": float(spec.lambda_cons),
                "LAMBDA_PRIOR": float(spec.lambda_prior),

                "TRAINING_STRATEGY": str(args.strategy).strip().lower(),
                "FAST_SENSITIVITY": bool(args.fast),
                "DISABLE_EARLY_STOPPING": bool(args.no_early_stopping),

                # big speed win: don’t reload inference model from disk
                "USE_IN_MEMORY_MODEL": True,

                # keep your "grid hygiene" defaults:
                "AUDIT_STAGES": "off",
                "DEBUG": False,
                "LOG_Q_DIAGNOSTICS": False,

                # keep your previous driver behavior
                "MV_WEIGHT": 0.0,
            }

            # Optional knobs from CLI (mirror make_env behavior)
            if args.disable_q:
                overrides["Q_POLICY_DATA_FIRST"] = "always_off"
                overrides["Q_POLICY_PHYSICS_FIRST"] = "always_off"
                overrides["LAMBDA_Q"] = 0.0
                overrides["LAMBDA_Q_DATA_FIRST"] = 0.0
                overrides["LAMBDA_Q_PHYSICS_FIRST"] = 0.0
            elif args.lambda_q is not None:
                overrides["LAMBDA_Q"] = float(args.lambda_q)
                overrides["LAMBDA_Q_DATA_FIRST"] = float(args.lambda_q)
                overrides["LAMBDA_Q_PHYSICS_FIRST"] = float(args.lambda_q)

            if args.disable_subs_resid:
                overrides["SUBS_RESID_POLICY_DATA_FIRST"] = "always_off"
                overrides["SUBS_RESID_POLICY_PHYSICS_FIRST"] = "always_off"
                overrides["ALLOW_SUBS_RESIDUAL"] = False

            if args.no_physics_ramp:
                overrides["PHYSICS_WARMUP_STEPS"] = 0
                overrides["PHYSICS_RAMP_STEPS"] = 0
            else:
                if args.physics_warmup_steps is not None:
                    overrides["PHYSICS_WARMUP_STEPS"] = int(args.physics_warmup_steps)
                if args.physics_ramp_steps is not None:
                    overrides["PHYSICS_RAMP_STEPS"] = int(args.physics_ramp_steps)

            if args.lambda_gw is not None:
                overrides["LAMBDA_GW"] = float(args.lambda_gw)
            if args.lambda_smooth is not None:
                overrides["LAMBDA_SMOOTH"] = float(args.lambda_smooth)
            if args.lambda_bounds is not None:
                overrides["LAMBDA_BOUNDS"] = float(args.lambda_bounds)
            if args.lambda_mv is not None:
                overrides["LAMBDA_MV"] = float(args.lambda_mv)
            else:
                # keep the same behavior as your current driver
                overrides["LAMBDA_MV"] = 0.0

            if bool(args.dry_run):
                print("[DryRun GOLD] would run:", spec.run_tag())
                continue

            try:
                run_dir = run_one_gold(
                    ctx,
                    overrides=overrides,
                    run_tag=spec.run_tag(),
                    stable_run_dir=True,
                    eval_max_batches=args.eval_max_batches,
                    cal_max_batches=args.eval_max_batches,
                )

                # Mark done in-memory + state file
                k = spec.key()
                completed.add(k)
                last_done = k
                _save_state(
                    state_path,
                    city=city,
                    scan_root=scan_root,
                    completed_n=len(completed),
                    last_key=last_done,
                )

                cleanup_between_runs()

                print("[GOLD] run_dir ->", run_dir)

            except Exception as e:
                msg = f"failed: {spec.tag()} ({type(e).__name__}: {e})"
                failures.append((i, msg))
                print("[Sensitivity GOLD] ERROR:", msg)
                if not args.continue_on_error:
                    raise

        print("\n[Sensitivity GOLD] Finished.")
        if failures:
            print("[Sensitivity GOLD] Failures:")
            for _, msg in failures:
                print("  - " + msg)
            raise SystemExit(1)

        return  # IMPORTANT: don’t fall through to old runner
    
    nj = resolve_n_jobs(args.n_jobs)
    
    if dev == "gpu":
        # Single GPU => force n_jobs=1
        if len(gpus) <= 1 and nj > 1:
            print("[Warn] Single GPU detected.")
            print("       Forcing --n-jobs 1.")
            nj = 1
    
        # Multi GPU => cap workers to num GPUs (safe)
        if len(gpus) >= 2:
            if nj > len(gpus):
                print("[Warn] Capping jobs to GPUs.")
                nj = len(gpus)
            
    if nj > 1 and (args.gold or args.inprocess):
        print(
            "[Warn] --n-jobs ignored with "
            "--gold/--inprocess."
        )
        nj = 1
    
    if nj > 1:
        def _worker(i: int, spec: RunSpec) -> str:
            env0 = make_env(
                base_env,
                epochs=args.epochs,
                spec=spec,
                strategy=args.strategy,
                disable_q=bool(args.disable_q),
                disable_subs_resid=bool(
                    args.disable_subs_resid
                ),
                no_physics_ramp=bool(
                    args.no_physics_ramp
                ),
                physics_warmup_steps=(
                    args.physics_warmup_steps
                ),
                physics_ramp_steps=(
                    args.physics_ramp_steps
                ),
                lambda_gw=args.lambda_gw,
                lambda_smooth=args.lambda_smooth,
                lambda_bounds=args.lambda_bounds,
                lambda_mv=args.lambda_mv,
                lambda_q=args.lambda_q,
                no_early_stopping=bool(
                    args.no_early_stopping
                ),
                fast=bool(args.fast),
                eval_max_batches=(
                    args.eval_max_batches
                ),
                batch_size=args.batch_size
            )
    
            env1 = apply_thread_env(
                env0,
                n_jobs=nj,
                threads=int(args.threads or 0),
            )
    
            if dev == "gpu":
                gid = pick_gpu_id(i, gpus)
                env1 = apply_gpu_env(
                    env1,
                    gpu_id=gid,
                    allow_growth=bool(
                        args.gpu_allow_growth
                    ),
                )
            else:
                gid = None
    
            _worker_banner(
                mode="Sensitivity",
                job_i=i,
                n_jobs=len(grid),
                pool=nj,
                run_tag=spec.run_tag(),
                device=dev,
                gpu_id=gid,
            )

            run_one_script(
                train_script,
                env=env1,
                dry_run=bool(args.dry_run),
                inprocess=False,
            )
            return spec.key()
    
        failures = []
        with ThreadPoolExecutor(max_workers=nj) as ex:
            futs = {
                ex.submit(_worker, i, s): (i, s)
                for i, s in enumerate(grid)
            }
            
            for fut in as_completed(futs):
                i, spec = futs[fut]
                try:
                    k = fut.result()
                    completed.add(k)
                    _save_state(
                        state_path,
                        city=city,
                        scan_root=scan_root,
                        completed_n=len(completed),
                        last_key=k,
                    )
                except Exception as e:
                    msg = f"failed: worker={i} {spec.tag()} ({e})"
                    failures.append(msg)
                    print("[Sensitivity] ERROR:", msg)
                    if not args.continue_on_error:
                        raise
    
        if failures:
            raise SystemExit(1)
    
        return

    failures: List[Tuple[int, str]] = []
    last_done: Optional[str] = None
    
    for i, spec in enumerate(grid):
        tag = spec.tag()
        print("\n" + "=" * 62)
        print(f"[Sensitivity] Run {i+1}/{len(grid)}")
        print(f"  {tag}")
        print("=" * 62)

        env = make_env(
            base_env,
            epochs=args.epochs,
            spec=spec,
            strategy=args.strategy,
            disable_q=bool(args.disable_q),
            disable_subs_resid=bool(args.disable_subs_resid),
            no_physics_ramp=bool(args.no_physics_ramp),
            physics_warmup_steps=args.physics_warmup_steps,
            physics_ramp_steps=args.physics_ramp_steps,
            lambda_gw=args.lambda_gw,
            lambda_smooth=args.lambda_smooth,
            lambda_bounds=args.lambda_bounds,
            lambda_mv=args.lambda_mv,
            lambda_q=args.lambda_q,
            no_early_stopping=bool(
                args.no_early_stopping
            ),
            fast=bool(args.fast),
            eval_max_batches=args.eval_max_batches,
            batch_size = args.batch_size 
        )

        gid = None
        if dev == "gpu":
            # sequential case: pick first visible GPU for clarity
            gid = pick_gpu_id(0, gpus)
        
        _worker_banner(
            mode="Sensitivity",
            job_i=i,
            n_jobs=len(grid),
            pool=1,
            run_tag=spec.run_tag(),
            device=dev,
            gpu_id=gid,
        )

        try:
            run_one_script(
                train_script,
                env=env,
                dry_run=bool(args.dry_run),
                inprocess=bool(args.inprocess),
            )
            # Mark done in-memory (useful if rerun same process)
            k = spec.key()
            completed.add(k)
            last_done = k
            _save_state(
                state_path,
                city=city,
                scan_root=scan_root,
                completed_n=len(completed),
                last_key=last_done,
            )
        except subprocess.CalledProcessError as e:
            msg = f"failed: {tag} (code={e.returncode})"
            failures.append((i, msg))
            print("[Sensitivity] ERROR: " + msg)
            if not args.continue_on_error:
                _save_state(
                    state_path,
                    city=city,
                    scan_root=scan_root,
                    completed_n=len(completed),
                    last_key=last_done,
                )
                raise

    print("\n[Sensitivity] Finished.")
    if failures:
        print("[Sensitivity] Failures:")
        for _, msg in failures:
            print("  - " + msg)
        raise SystemExit(1)

    print(
        "You can now run make_supp_figS6_ablations.py "
        "over the same --root to build the tidy table "
        "+ Supplement S6 figure."
    )

        
if __name__ == "__main__":
    main()
