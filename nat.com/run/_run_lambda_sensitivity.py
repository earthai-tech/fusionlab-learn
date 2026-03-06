# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>
"""
run_lambda_sensitivity.py

Small driver to run a (lambda_cons, lambda_prior) sensitivity grid for
GeoPriorSubsNet using the existing Stage-2 training script.

It simply calls `training_NATCOM_GEOPRIOR.py` multiple times with
environment overrides:

    - EPOCHS_OVERRIDE
    - PDE_MODE_OVERRIDE
    - LAMBDA_CONS_OVERRIDE
    - LAMBDA_PRIOR_OVERRIDE

Each run writes its own `ablation_records/ablation_record.jsonl` entry.
Your `make_supp_figS6_ablations.py` will then pick up all those records.

Usage (example)
---------------
# For Nansha:
set CITY=nansha
python nat.com/run_lambda_sensitivity.py --epochs 20

# For Zhongshan:
set CITY=zhongshan
python nat.com/run_lambda_sensitivity.py --epochs 20
"""

from __future__ import annotations

import argparse
import itertools
import os
import subprocess
import sys
from pathlib import Path


# Default grids
DEFAULT_LCONS  = [1.0] # [0.01, 0.05, 0.10, 0.20]
DEFAULT_LPRIOR = [0.5] # [0.01, 0.05, 0.10, 0.20]
DEFAULT_PDE_MODES =["both"] #  ["none", "both"]  # you can trim to ["both"] if you want

TRAIN_SCRIPT = Path(__file__).with_name("stage2.py") 



def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run lambda_cons / lambda_prior sensitivity grid "
                    "using stage2.py (training)."
    )
    p.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Epochs per sensitivity run (short runs).",
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
        "--pde-modes",
        type=str,
        nargs="+",
        default=DEFAULT_PDE_MODES,
        help="Which PDE_MODE_CONFIG values to sweep "
             "(e.g. none both).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not TRAIN_SCRIPT.exists():
        raise SystemExit(
            f"Cannot find training script at {TRAIN_SCRIPT}. "
            "Adjust TRAIN_SCRIPT path in run_lambda_sensitivity.py."
        )

    base_env = os.environ.copy()
    city = base_env.get("CITY", "<unknown>")
    print(f"[Sensitivity] CITY={city}, epochs={args.epochs}")
    print(f"  lambda_cons  grid: {args.lcons}")
    print(f"  lambda_prior grid: {args.lprior}")
    print(f"  PDE modes        : {args.pde_modes}")

    # Cartesian product over PDE mode, lambda_cons, lambda_prior
    for pde_mode in args.pde_modes:
        for ii, (lc, lp) in enumerate (itertools.product(args.lcons, args.lprior)):
            
            # if ii < 14: 
            #     continue 
            
            tag = f"pde={pde_mode}, lcons={lc:g}, lprior={lp:g}"
            print("\n" + "=" * 72)
            print(f"[Sensitivity] Running {tag}")
            print("=" * 72)

            env = base_env.copy()
            env["PDE_MODE_OVERRIDE"]   = pde_mode
            env["EPOCHS_OVERRIDE"]     = str(args.epochs)
            env["LAMBDA_CONS_OVERRIDE"]   = str(lc)
            env["LAMBDA_PRIOR_OVERRIDE"]  = str(lp)

            # (Optional) ensure GW / smooth / mv stay fixed:
            # env["LAMBDA_GW_OVERRIDE"]     = "0.01"
            # env["LAMBDA_SMOOTH_OVERRIDE"] = "0.01"
            # env["LAMBDA_MV_OVERRIDE"]     = "0.01"

            cmd = [sys.executable, str(TRAIN_SCRIPT)]
            # Will raise if a run fails; you can add try/except if you
            # prefer to continue on error.
            subprocess.run(cmd, env=env, check=True)

    print("\n[Sensitivity] All combinations finished.\n"
          "You can now run make_supp_figS6_ablations.py over the same "
          "--root to build the tidy table + Supplement S6 figure.")


if __name__ == "__main__":
    main()
