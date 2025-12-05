# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>
#
# High-level training workflow for the GeoPrior GUI.
#
# This module centralises all "smart" training logic that used to live
# inside GeoPriorForecaster._on_train_clicked and
# _run_train_dry_preview.
#
# Responsibilities
# ----------------
# - Validate GUI inputs (city, CSV, experiment name) in a UI-agnostic way.
# - Update GeoPriorConfig (TRAIN_CSV_PATH, CITY_NAME, EXPERIMENT_NAME).
# - Call cfg.ensure_valid().
# - Ask Stage1Service whether Stage-1 should be (re)run or reused.
# - Build cfg_overrides (including device overrides) via BaseWorkflow.
# - Provide:
#       * a dry-run preview (no threads, no I/O),
#       * a real run planner that calls back into the GUI to start
#         Stage-1 and/or Training threads.


from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from ..services.stage1_service import Stage1Service, Stage1Decision
from .base import BaseWorkflowController, RunEnv, GUIHooks



# ----------------------------------------------------------------------
# Small GUI state/data objects
# ----------------------------------------------------------------------


@dataclass
class TrainGuiState:
    """
    Minimal snapshot of what the Train tab provides.

    Parameters
    ----------
    city_text : str
        Raw text from the City/Dataset field.
    csv_path : Path or None
        Path to the selected training CSV, or ``None`` if nothing has
        been chosen yet.
    experiment_name : str or None
        Optional experiment name typed by the user. If ``None`` or
        empty, a default will be derived from the config.
    """

    city_text: str
    csv_path: Optional[Path]
    experiment_name: Optional[str]


@dataclass
class TrainPlan:
    """
    Fully resolved training plan.

    This object is produced by :meth:`TrainController.plan_from_gui`
    and can be used both for dry-run previews and real runs.

    Parameters
    ----------
    city : str
        Effective city name that will be used for training.
    csv_path : str
        Path to the training CSV as a normalised string.
    cfg_overrides : dict
        Configuration overrides to pass to backend runners
        (Stage-1 / training). This is produced via
        :meth:`BaseWorkflowController.build_cfg_overrides`.
    stage1_decision : Stage1Decision
        Result of the smart Stage-1 handshake.
    experiment_name : str
        Experiment name that will be set on the config.
    """

    city: str
    csv_path: str
    cfg_overrides: Dict[str, Any]
    stage1_decision: Stage1Decision
    experiment_name: str


# ----------------------------------------------------------------------
# Controller
# ----------------------------------------------------------------------


class TrainController(BaseWorkflowController):
    """
    High-level orchestrator for the training workflow.

    This controller is **UI-agnostic**: it only depends on:

    - :class:`RunEnv` for the current GeoPriorConfig, runs root, etc.;
    - :class:`GUIHooks` for logging/status/progress/error;
    - :class:`Stage1Service` for Stage-1 reuse decisions.

    The GeoPriorForecaster GUI is responsible for:

    - building a :class:`TrainGuiState` from widgets;
    - calling :meth:`dry_preview` when in dry mode;
    - calling :meth:`start_real_run` otherwise and providing callbacks
      that actually start Stage1Thread / TrainingThread.
    """

    def __init__(
        self,
        env: RunEnv,
        hooks: GUIHooks,
        stage1_svc: Stage1Service,
    ) -> None:
        super().__init__(env, hooks)
        self.stage1_svc = stage1_svc

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _infer_city_from_path(path: Path) -> str:
        """
        Infer a city/dataset name from a CSV filename.

        Examples
        --------
        nansha.csv       -> "nansha"
        my city file.csv -> "my_city_file"
        """
        stem = path.stem.strip()
        if not stem:
            return "geoprior_city"
        return stem.replace(" ", "_")

    def _derive_experiment_name(
        self,
        cfg: Any,
        requested: Optional[str],
    ) -> str:
        """
        Decide which experiment name to use.

        Priority:
        1) explicit name from the GUI (TrainGuiState.experiment_name),
        2) existing cfg.EXPERIMENT_NAME if set,
        3) CITY_NAME,
        4) "geoprior_run".
        """
        if requested:
            name = requested.strip()
            if name:
                return name

        existing = getattr(cfg, "EXPERIMENT_NAME", "") or ""
        if existing:
            return str(existing)

        city = getattr(cfg, "CITY_NAME", "") or ""
        if city:
            return str(city)

        return "geoprior_run"

    # ------------------------------------------------------------------
    # Core planning
    # ------------------------------------------------------------------
    def plan_from_gui(
        self,
        gui_state: TrainGuiState,
    ) -> Optional[TrainPlan]:
        """
        Build a TrainPlan from the current GUI state.

        This method performs the following steps:

        1) Validate that a CSV is selected.
        2) Derive an effective city name from the GUI text or CSV stem.
        3) Update GeoPriorConfig with TRAIN_CSV_PATH, CITY_NAME,
           EXPERIMENT_NAME.
        4) Call cfg.ensure_valid().
        5) Ask Stage1Service whether Stage-1 should be (re)run or reused.
        6) Build cfg_overrides via BaseWorkflowController.

        Parameters
        ----------
        gui_state : TrainGuiState
            Current Train tab inputs.

        Returns
        -------
        TrainPlan or None
            A TrainPlan if validation succeeded, otherwise ``None`` if
            a fatal validation error occurred (and a GUI message has
            already been emitted via hooks.warn/error).
        """
        cfg = self.env.geo_cfg

        # 1) Validate CSV
        if gui_state.csv_path is None:
            self.hooks.warn(
                "No training dataset",
                "Please choose a training data file first.",
            )
            self.status("Train: aborted (no training data).")
            self.progress(0.0)
            return None

        csv_path = gui_state.csv_path
        csv_path_str = str(csv_path)

        # 2) Derive effective city name
        city = (gui_state.city_text or "").strip()
        if not city:
            city = self._infer_city_from_path(csv_path)

        if not city:
            self.hooks.warn(
                "Missing city name",
                "Please provide a city/dataset name.",
            )
            self.status("Train: aborted (missing city name).")
            self.progress(0.0)
            return None

        # 3) Update config with basic fields
        cfg.TRAIN_CSV_PATH = csv_path_str

        if not getattr(cfg, "CITY_NAME", ""):
            cfg.CITY_NAME = city

        experiment_name = self._derive_experiment_name(
            cfg=cfg,
            requested=gui_state.experiment_name,
        )
        cfg.EXPERIMENT_NAME = experiment_name

        # 4) Validate configuration
        try:
            cfg.ensure_valid()
        except Exception as exc:  # pragma: no cover - pure GUI validation
            self.hooks.error(
                "Invalid configuration",
                f"The training configuration is not valid:\n\n{exc}",
            )
            self.status("Train: aborted (invalid configuration).")
            self.progress(0.0)
            return None

        # 5) Compute Stage-1 decision via the service
        clean_flag = bool(getattr(cfg, "clean_stage1_dir", False))
        stage1_decision = self.stage1_svc.decide(
            city=city,
            clean_stage1_dir=clean_flag,
        )

        # 6) Build cfg_overrides (GeoPriorConfig + device overrides)
        cfg_overrides = self.build_cfg_overrides()
        # Ensure BASE_OUTPUT_DIR is rooted under the GUI runs root
        cfg_overrides.setdefault(
            "BASE_OUTPUT_DIR",
            str(self.env.gui_runs_root),
        )

        return TrainPlan(
            city=city,
            csv_path=csv_path_str,
            cfg_overrides=cfg_overrides,
            stage1_decision=stage1_decision,
            experiment_name=experiment_name,
        )

    # ------------------------------------------------------------------
    # Public entrypoints
    # ------------------------------------------------------------------
    def dry_preview(self, gui_state: TrainGuiState) -> None:
        """
        Dry-run preview of the training workflow (no threads, no I/O).

        Steps
        -----
        1) Build a TrainPlan via :meth:`plan_from_gui`.
        2) If validation failed → return.
        3) If Stage-1 handshake was cancelled → log & return.
        4) Log a human-readable summary of what *would* happen.
        5) Drive the progress bar from 0 → 1.0 using GUIHooks.
        """
        self.progress(0.0)
        self.status("Dry-run / Train: validating inputs…")

        plan = self.plan_from_gui(gui_state)
        if plan is None:
            # Validation error already reported
            return

        dec = plan.stage1_decision
        cfg = self.env.geo_cfg
        root = self.env.gui_runs_root

        # Mid progress: config & Stage-1 decision ok
        self.progress(0.4)

        # 4) Log the planned workflow summary
        lines = [
            "[Dry-run / Train] Planned workflow:",
            f"  City           : {getattr(cfg, 'CITY_NAME', plan.city)}",
            f"  CSV            : {plan.csv_path}",
            f"  Results root   : {root}",
        ]

        # These attributes exist on GeoPriorConfig in your current code,
        # but we guard with getattr for robustness.
        time_steps = getattr(cfg, "time_steps", None)
        horizon = getattr(cfg, "forecast_horizon_years", None)
        epochs = getattr(cfg, "epochs", None)
        batch_size = getattr(cfg, "batch_size", None)
        pde_mode = getattr(cfg, "pde_mode", None)
        clean_stage1 = getattr(cfg, "clean_stage1_dir", None)
        build_future = getattr(cfg, "build_future_npz", None)

        if time_steps is not None:
            lines.append(f"  Time steps     : {time_steps}")
        if horizon is not None:
            lines.append(f"  Horizon (years): {horizon}")
        if epochs is not None or batch_size is not None:
            lines.append(
                f"  Epochs / batch : {epochs} / {batch_size}"
            )
        if pde_mode is not None:
            lines.append(f"  PDE mode       : {pde_mode}")
        if clean_stage1 is not None:
            lines.append(f"  Clean Stage-1  : {clean_stage1}")
        if build_future is not None:
            lines.append(f"  Build future   : {build_future}")
        if plan.experiment_name:
            lines.append(f"  Experiment     : {plan.experiment_name}")

        for line in lines:
            self.log(line)

        self.progress(0.55)

        # 5) Interpret Stage-1 decision
        if dec.cancelled:
            self.log(
                "[Dry-run] Result: user would cancel at the "
                "Stage-1 handshake → no training would run."
            )
            self.progress(1.0)
            self.status(
                "Dry-run / Train: preview complete (100 %) – "
                "user would cancel at Stage-1."
            )
            return

        if dec.need_stage1:
            # Stage1Service already logged detailed reasoning.
            self.log(
                "[Dry-run] Result: Stage-1 would be (re)built "
                "before Stage-2 training."
            )
        else:
            manifest = dec.manifest_hint or "<unknown>"
            self.log(
                "[Dry-run] Result: Stage-1 would be reused.\n"
                f"  manifest: {manifest}"
            )

        self.progress(0.8)

        # 6) Final message: Stage-2 training that *would* run
        self.log(
            "[Dry-run] After Stage-1, a TrainingThread would be started "
            "with the above configuration (but in dry mode, "
            "nothing is run)."
        )

        self.progress(1.0)
        self.status("Dry-run / Train: preview complete (100 %).")

    def start_real_run(
        self,
        gui_state: TrainGuiState,
        start_stage1_cb: Callable[[str, Dict[str, Any]], None],
        start_training_cb: Callable[[str, Dict[str, Any]], None],
    ) -> None:
        """
        Start a real training run, delegating thread creation to the GUI.

        Parameters
        ----------
        gui_state : TrainGuiState
            Current Train tab inputs.
        start_stage1_cb : callable
            Callback ``f(city: str, cfg_overrides: dict) -> None``
            used to launch Stage-1. The GUI typically assigns
            ``self._cfg_overrides = cfg_overrides`` and then calls
            ``self._start_stage1(city)``.
        start_training_cb : callable
            Callback ``f(manifest_path: str, cfg_overrides: dict) -> None``
            used to launch Training. The GUI typically assigns
            ``self._cfg_overrides = cfg_overrides`` and then calls
            ``self._start_training(manifest_path)``.
        """
        self.progress(0.0)
        self.status("Train: preparing workflow…")

        plan = self.plan_from_gui(gui_state)
        if plan is None:
            # Validation error already reported
            return

        dec = plan.stage1_decision

        if dec.cancelled:
            self.status("Training cancelled before Stage-1.")
            self.progress(0.0)
            return

        # At this point the config is valid and Stage-1 decision is known.
        if dec.need_stage1:
            # Stage-1 must be (re)run; Training will be triggered when
            # Stage-1 completes and emits a manifest.
            self.status("Stage-1: preparing sequences…")
            start_stage1_cb(plan.city, plan.cfg_overrides)
        else:
            # We can jump straight to Stage-2 training using the chosen
            # manifest path.
            manifest = dec.manifest_hint
            if not manifest:
                # Very defensive: Stage1Service should always provide a
                # manifest hint when need_stage1 is False.
                self.hooks.error(
                    "Missing Stage-1 manifest",
                    "Stage-1 decision indicated reuse but did not "
                    "provide a manifest path.\n\n"
                    "Please rebuild Stage-1 for this city.",
                )
                self.status("Training aborted (no Stage-1 manifest).")
                self.progress(0.0)
                return

            self.status("Stage-2: training GeoPrior model (reusing Stage-1)…")
            start_training_cb(manifest, plan.cfg_overrides)
