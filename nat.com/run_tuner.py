#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_tuner.py

Code Ocean-friendly driver:
1) Runs Stage 1 data prep:   main_NATCOM_stage1_prepared.py
2) Then runs tuner stage:    tune_NATCOM_GEOPRIOR.py

Usage
-----
$ python run_tuner.py \
    --stage1 main_NATCOM_stage1_prepared.py \
    --next   tune_NATCOM_GEOPRIOR.py \
    --skip-stage1  # (optional) if Stage-1 already ran
"""

import argparse
import os
import sys
import time
import glob
import subprocess
from datetime import datetime
from pathlib import Path

DEFAULT_STAGE1 = "main_NATCOM_stage1_prepared.py"
DEFAULT_NEXT   = "tune_NATCOM_GEOPRIOR.py"
LOG_DIR        = Path("results/_logs")


def _ts() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def run_python(script: str, extra_args=None, env=None, log_path: Path | None = None) -> int:
    cmd = [sys.executable, "-u", script] + (extra_args or [])
    proc_env = os.environ.copy()
    proc_env["PYTHONUNBUFFERED"] = "1"
    if env:
        proc_env.update(env)

    print(f"\n▶ Running: {' '.join(cmd)}")
    print(f"   Working dir: {os.getcwd()}")

    lf = None
    try:
        if log_path:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            lf = open(log_path, "a", encoding="utf-8")
            lf.write(f"\n===== START {script} @ {time.ctime()} =====\n")

        with subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=proc_env,
            bufsize=1,
            universal_newlines=True,
        ) as p:
            for line in p.stdout:
                line = line.rstrip("\n")
                print(line)
                if lf:
                    lf.write(line + "\n")
            rc = p.wait()

        if lf:
            lf.write(f"===== END {script} (rc={rc}) @ {time.ctime()} =====\n")
        return rc
    finally:
        if lf:
            lf.flush()
            lf.close()


def find_latest_manifest(
        pattern: str = "results/*_GeoPriorSubsNet_stage1/manifest.json"
    ) -> str | None:
    matches = glob.glob(pattern)
    if not matches:
        return None
    matches.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return matches[0]


def main():
    parser = argparse.ArgumentParser(description="Run Stage-1 then Tuner.")
    parser.add_argument("--stage1", default=DEFAULT_STAGE1,
                        help="Path to Stage-1 script.")
    parser.add_argument("--next", default=DEFAULT_NEXT,
                        help="Path to tuner script.")
    parser.add_argument("--skip-stage1", action="store_true",
                        help="Skip Stage-1 (useful if it already ran).")
    parser.add_argument("--manifest", default=None,
                        help="Path to Stage-1 manifest.json (optional).")
    args, unknown = parser.parse_known_args()

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    tuner_log = LOG_DIR / f"tuner-{_ts()}.log"

    # 1) Stage-1
    if not args.skip_stage1:
        rc = run_python(args.stage1, env=None, log_path=tuner_log)
        if rc != 0:
            print(f"\n XXX Stage-1 failed with return code {rc}. Aborting.")
            sys.exit(rc)
    else:
        print("\n III Skipping Stage-1 as requested.")

    # 2) Discover manifest (if not provided)
    manifest_path = args.manifest or find_latest_manifest()
    if manifest_path:
        print(f"\n Found Stage-1 manifest: {manifest_path}")
    else:
        print("\n ^^^ No Stage-1 manifest found. Proceeding without STAGE1_MANIFEST.")
    env = {"STAGE1_MANIFEST": manifest_path} if manifest_path else None

    # 3) Tuner
    rc = run_python(args.next, extra_args=unknown, env=env, log_path=tuner_log)
    if rc != 0:
        print(f"\n XXX Tuner failed with return code {rc}.")
        sys.exit(rc)

    print("\n  All done. Logs saved to:", tuner_log)


if __name__ == "__main__":
    main()
