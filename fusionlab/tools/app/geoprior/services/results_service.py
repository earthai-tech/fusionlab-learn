# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>
#
# Small utility service to:
#   - discover the latest transferability run under a results root;
#   - persist the GUI log cache into a run directory.
#
# This keeps GeoPriorForecaster free from low-level filesystem logic.

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Optional

from .xfer_view import latest_xfer_csv, latest_xfer_json

class ResultsService:
    """
    Pure helper for results-related tasks.

    Parameters
    ----------
    log : callable
        Logging function ``f(msg: str) -> None`` used for
        optional debug messages.
    """

    def __init__(self, log: Callable[[str], None]) -> None:
        self.log = log

    # ------------------------------------------------------------------
    # Transferability helpers
    # ------------------------------------------------------------------
    def discover_last_xfer(self, results_root: Path) -> Dict[str, Any]:
        """
        Best-effort discovery of the latest transferability run
        under ``results_root``.

        Returns
        -------
        dict
            Either an empty dict (if nothing found) or a mapping with:
                - "out_dir"   : str, run directory
                - "csv_path"  : str or None
                - "json_path" : str or None
        """
        out: Dict[str, Any] = {}

        root_str = str(results_root).strip()
        if not root_str:
            # Nothing to search
            self.log(
                "[ResultsService] No xfer results root provided – "
                "skipping discovery."
            )
            return out

        # latest_xfer_* take strings, not Path
        csv_path = latest_xfer_csv(root_str)
        json_path = latest_xfer_json(root_str)

        if not csv_path and not json_path:
            self.log(
                "[ResultsService] No transferability artifacts found under:\n"
                f"  {root_str}"
            )
            return out

        best_path = csv_path or json_path
        run_dir = Path(best_path).parent

        out = {
            "out_dir": str(run_dir),
            "csv_path": str(csv_path) if csv_path else None,
            "json_path": str(json_path) if json_path else None,
        }

        self.log(
            "[ResultsService] Detected latest transferability run:\n"
            f"  {run_dir}"
        )
        return out

    # ------------------------------------------------------------------
    # GUI log persistence
    # ------------------------------------------------------------------
    def save_gui_log(
        self,
        log_mgr: Optional[Any],
        result: Dict[str, Any],
    ) -> None:
        """
        Persist the current GUI log cache into the run's directory.

        Parameters
        ----------
        log_mgr : object or None
            Object exposing ``save_cache(run_dir) -> str`` or ``None``
            if logging is disabled / not available.
        result : dict
            Result mapping returned by training / tuning / inference /
            xfer, expected to contain one of:
                - 'run_dir'
                - 'run_output_path'
                - 'out_dir'
        """
        if log_mgr is None:
            # Nothing to do; caller doesn't have a log manager
            return

        run_dir = (
            result.get("run_dir")
            or result.get("run_output_path")
            or result.get("out_dir")
        )
        if not run_dir:
            return

        try:
            log_path = log_mgr.save_cache(run_dir)
            self.log(f"GUI log saved to:\n  {log_path}")
        except Exception as exc:
            self.log(f"[Warn] Could not save GUI log file: {exc}")


