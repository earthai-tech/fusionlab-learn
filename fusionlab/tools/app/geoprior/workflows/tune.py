# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>
#
# Workflow controller for Stage-2 hyperparameter tuning.
#
# This module mirrors the pattern used for training:
#
#   - TuneGuiState:  minimal data coming from the GUI
#   - TunePlan:      validated + enriched plan used by back-end threads
#   - TuneController: pure, Qt-free logic for real runs and dry preview
#
# The goal is to keep the GeoPrior GUI thin:
#   * resolve TuneJobSpec / QuickTuneDialog in the GUI layer,
#   * build a TuneGuiState object,
#   * delegate all tuning workflow decisions to TuneController.

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Callable

from ..config import default_tuner_search_space
from .base import BaseWorkflowController, RunEnv, GUIHooks


# ----------------------------------------------------------------------
# Data classes
# ----------------------------------------------------------------------


@dataclass
class TuneGuiState:
    """
    Minimal snapshot of what the GUI provides for tuning.

    Parameters
    ----------
    city_text : str or None
        City / dataset name currently in the City field. May be empty
        if the GUI is relying on a TuneJobSpec resolved elsewhere.
    manifest_path : str or None
        Explicit Stage-1 manifest path, if the GUI resolved a job
        (e.g. via TuneOptionsDialog or QuickTuneDialog). If ``None``,
        the back-end may auto-discover Stage-1 based on CITY_NAME and
        BASE_OUTPUT_DIR.
    stage1_root : str or None
        Optional Stage-1 run directory used only for logging in
        dry-run previews. Can be ``None``.
    tuner_search_space : dict or None
        Search space built from the Tune tab widgets (typically via
        ``_build_tuner_space_from_ui`` in the GUI). If ``None``,
        :func:`default_tuner_search_space` will be used as a fallback.
    max_trials : int or None
        Desired maximum number of trials. If ``None``, falls back to
        ``geo_cfg.TUNER_MAX_TRIALS`` or 20.
    eval_tuned : bool
        Whether the tuned model should be evaluated after search
        (mirrors the GUI checkbox).
    """

    city_text: Optional[str]
    manifest_path: Optional[str]
    stage1_root: Optional[str]
    tuner_search_space: Optional[Dict[str, Any]]
    max_trials: Optional[int]
    eval_tuned: bool


@dataclass
class TunePlan:
    """
    Fully validated tuning plan used by the back-end.

    Parameters
    ----------
    city : str
        Target city / dataset to tune for.
    manifest_path : str or None
        Stage-1 manifest to use. If ``None``, back-end may auto-discover
        based on CITY_NAME and BASE_OUTPUT_DIR.
    stage1_root : str or None
        Optional Stage-1 run directory (for logging / UI only).
    cfg_overrides : dict
        NAT-style configuration overrides passed to :func:`run_tuning`
        or :class:`TuningThread`.
    max_trials : int
        Final maximum number of tuning trials.
    """

    city: str
    manifest_path: Optional[str]
    stage1_root: Optional[str]
    cfg_overrides: Dict[str, Any]
    max_trials: int


# ----------------------------------------------------------------------
# Controller
# ----------------------------------------------------------------------


class TuneController(BaseWorkflowController):
    """
    Encapsulate Stage-2 tuning workflow (real + dry preview).

    This controller is intentionally Qt-free; all GUI interaction
    happens via GUIHooks and the GeoPriorForecaster keeps the
    widget-level concerns (TuneOptionsDialog, QuickTuneDialog, etc.).
    """

    def __init__(self, env: RunEnv, hooks: GUIHooks):
        super().__init__(env, hooks)

    # ------------------------------------------------------------------
    # Plan construction
    # ------------------------------------------------------------------
    def plan_from_gui(self, gui_state: TuneGuiState) -> Optional[TunePlan]:
        """
        Build and validate a TunePlan from the current GUI state.

        Steps
        -----
        - Validate that a city name is available.
        - Determine or fallback the tuner search space.
        - Determine max_trials from GUI or config.
        - Inject search space and CITY_NAME into geo_cfg.
        - Build cfg_overrides (including TUNER_MAX_TRIALS, CITY_NAME,
          BASE_OUTPUT_DIR, and possibly TUNER_SEARCH_SPACE).
        """

        # 1) City validation
        raw_city = (gui_state.city_text or "").strip()
        if not raw_city:
            # Let the GUI decide how to display this (QMessageBox, etc.)
            self.hooks.warn(
                "Missing city",
                "Please provide a city/dataset name, or pick one "
                "from the tuning options.",
            )
            return None

        city = raw_city

        # 2) Resolve search space
        search_space = gui_state.tuner_search_space
        if not search_space:
            # Fall back to default search space from config
            search_space = default_tuner_search_space()

        cfg = self.env.geo_cfg

        # Let the config carry the search space if it exposes the field
        if hasattr(cfg, "tuner_search_space"):
            cfg.tuner_search_space = search_space

        # 3) Determine max_trials
        max_trials = gui_state.max_trials
        if max_trials is None:
            max_trials = int(getattr(cfg, "TUNER_MAX_TRIALS", 20))
        else:
            max_trials = int(max_trials)

        # 4) Build NAT-style overrides, including TUNER_SEARCH_SPACE
        overrides = cfg.to_cfg_overrides()
        if self.env.device_overrides:
            overrides.update(self.env.device_overrides)

        # Ensure search space is present in overrides if back-end expects it
        overrides.setdefault("TUNER_SEARCH_SPACE", search_space)

        # Inject desired max_trials into NAT config
        overrides["TUNER_MAX_TRIALS"] = max_trials

        # Force GUI runs under ~/.fusionlab_runs (or whatever gui_runs_root is)
        if getattr(self.env, "gui_runs_root", None) is not None:
            overrides.setdefault(
                "BASE_OUTPUT_DIR",
                str(Path(self.env.gui_runs_root)),
            )

        # Inject desired city (Stage-1 for this city must exist)
        overrides["CITY_NAME"] = city

        plan = TunePlan(
            city=city,
            manifest_path=gui_state.manifest_path,
            stage1_root=gui_state.stage1_root,
            cfg_overrides=overrides,
            max_trials=max_trials,
        )
        return plan

    # ------------------------------------------------------------------
    # Dry-run preview
    # ------------------------------------------------------------------
    def dry_preview(self, gui_state: TuneGuiState) -> None:
        """
        Simulate what a tuning run would do, without starting threads.

        This mirrors the behaviour of the old `_run_tune_dry_preview`,
        but lives in a pure, testable controller.

        The caller (GeoPriorForecaster) is responsible for:
        - clearing any log manager,
        - disabling / enabling buttons,
        - calling this method only when dry-mode is active.
        """
        # Start dry-run preview: reset progress & status
        self.progress(0.0)
        self.status("Dry-run / Tune: resolving tuning job…")

        self.log("[Dry-run] Previewing GeoPrior tuning workflow...")

        plan = self.plan_from_gui(gui_state)
        if plan is None:
            # plan_from_gui already warned the user
            self.progress(1.0)
            self.status(
                "Dry-run / Tune: preview aborted – invalid configuration."
            )
            return

        # Progress: plan resolved successfully
        self.progress(0.35)
        self.status(
            f"Dry-run / Tune: syncing configuration for city={plan.city}…"
        )

        # Use the search space from overrides or a default if missing
        search_space = plan.cfg_overrides.get(
            "TUNER_SEARCH_SPACE", default_tuner_search_space()
        )

        # Progress: configuration + search space ready
        self.progress(0.65)

        # Determine results_root used for this GUI
        results_root = Path(self.env.gui_runs_root)

        stage1_root = plan.stage1_root or "<from manifest / auto-discover>"
        manifest = plan.manifest_path or "<auto-discover>"

        # Log the tuning plan
        self.log(
            "[Dry-run] Tuning plan:\n"
            f"  city          : {plan.city}\n"
            f"  results_root  : {results_root}\n"
            f"  stage1_root   : {stage1_root}\n"
            f"  manifest      : {manifest}\n"
            f"  max_trials    : {plan.max_trials}\n"
            f"  search keys   : {sorted(search_space.keys())}"
        )

        self.log(
            "[Dry-run] A TuningThread would be created with the above "
            "configuration, but in dry mode nothing is started."
        )

        # Finalise: set progress to 100 % and emit completion status
        self.progress(1.0)
        self.status(
            "Dry-run / Tune: preview complete (100 %)."
        )

    # ------------------------------------------------------------------
    # Real tuning run
    # ------------------------------------------------------------------
    def start_real_run(
        self,
        gui_state: TuneGuiState,
        start_tuning_cb: Callable[[TunePlan, bool], None],
    ) -> None:
        """
        Entry point for real tuning (non-dry mode).

        Parameters
        ----------
        gui_state : TuneGuiState
            Snapshot of the GUI state for tuning.
        start_tuning_cb : callable
            Callback provided by the GUI to actually start the
            TuningThread.

            Expected signature::

                start_tuning_cb(plan: TunePlan, eval_tuned: bool) -> None

            where `eval_tuned` mirrors the GUI checkbox
            (run evaluation on the tuned model).
        """
        plan = self.plan_from_gui(gui_state)
        if plan is None:
            # plan_from_gui already warned the user
            return

        # Update progress / status like the original _on_tune_clicked
        self.log(f"Start GeoPrior tuning for city={plan.city!r}.")
        self.status(
            f"Stage-2: tuning GeoPrior model for city={plan.city}."
        )
        self.progress(0.0)

        # Delegate to the GUI-provided callback, which will:
        #   - build and start TuningThread,
        #   - wire Qt signals,
        #   - manage buttons, timers, etc.
        start_tuning_cb(plan, gui_state.eval_tuned)
