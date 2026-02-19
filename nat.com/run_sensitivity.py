# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>
"""
run_lambda_sensitivity.py

Driver to run a (lambda_cons, lambda_prior) sensitivity grid
for GeoPriorSubsNet using the existing Stage-2 training script.

This script calls stage2.py multiple times with environment
overrides. Each run should write its own ablation record
(entry in ablation_records/ablation_record.jsonl), which your
make_supp_figS6_ablations.py later aggregates.

Core overrides (expected by stage2.py)
-------------------------------------
- EPOCHS_OVERRIDE
- PDE_MODE_OVERRIDE
- LAMBDA_CONS_OVERRIDE
- LAMBDA_PRIOR_OVERRIDE

Optional "deconfounding" overrides (safe to export even if
stage2.py ignores some of them; you can wire them later)
-------------------------------------------------------
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
-----
set CITY=zhongshan
python nat.com/run_lambda_sensitivity.py --epochs 20


to force rerun everything:

python nat.com/run_lambda_sensitivity.py --epochs 20 --no-resume

results live elsewhere: 

python nat.com/run_lambda_sensitivity.py --epochs 20 \
  --scan-root F:/repositories/fusionlab-learn/results/zhongshan
    
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

from fusionlab.utils import default_results_dir

TRAIN_SCRIPT_DEFAULT = Path(__file__).with_name(
    "sensitivity.py"
)

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

DEFAULT_PDE_MODES: List[str] = ["none", "both"]


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
    except Exception:
        # State is optional: never fail the run.
        return


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
) -> Dict[str, str]:
    env = dict(base_env)

    # Core sweep
    env["PDE_MODE_OVERRIDE"] = str(spec.pde_mode)
    env["EPOCHS_OVERRIDE"] = str(int(epochs))
    env["LAMBDA_CONS_OVERRIDE"] = str(spec.lambda_cons)
    env["LAMBDA_PRIOR_OVERRIDE"] = str(spec.lambda_prior)

    # Traceability
    env["RUN_TAG"] = spec.run_tag()
    env["DISABLE_EARLY_STOPPING"] = "1"

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


def run_one(
    train_script: Path,
    *,
    env: Dict[str, str],
    dry_run: bool,
) -> None:
    cmd = [sys.executable, str(train_script)]
    if dry_run:
        print("[DryRun] " + " ".join(cmd))
        return
    subprocess.run(cmd, env=env, check=True)


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
        )

        try:
            run_one(
                train_script,
                env=env,
                dry_run=bool(args.dry_run),
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
