# geoprior/workflows/inference.py
# -*- coding: utf-8 -*-
# License: BSD-3-Clause
#
# Inference workflow controller for the GeoPrior GUI.
#
# This module centralises the "what should inference do?" logic in a
# testable, Qt-free class. The PyQt main window is responsible only
# for:
#
#   - reading widget values into an InferenceGuiState;
#   - starting / wiring InferenceThread via a callback;
#   - handling button states and timers.
#
# Behaviour is kept compatible with the original _on_infer_clicked and
# _run_infer_dry_preview implementations.

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from .base import BaseWorkflowController


# ----------------------------------------------------------------------
# Data structures
# ----------------------------------------------------------------------


@dataclass
class InferenceGuiState:
    """
    Snapshot of what the GUI provides for an inference run.

    Parameters
    ----------
    model_path : str
        Path to the trained/tuned .keras model.
    dataset_key : str
        One of {"train", "val", "test", "custom"} coming from the
        dataset combo box.
    use_future_npz : bool
        Whether to use the Stage-1 "future" NPZ instead of eval NPZ.
    manifest_path : str or None
        Explicit Stage-1 manifest path, if any. May be None to let the
        backend auto-discover it.
    inputs_npz : str or None
        Optional custom inputs .npz path (for dataset_key == "custom").
    targets_npz : str or None
        Optional custom targets .npz path (for dataset_key == "custom").
    cov_target : float
        Target coverage level (e.g. 0.80 for 80% intervals).
    include_gwl : bool
        Whether to include GWL outputs in inference.
    batch_size : int
        Batch size used for inference.
    make_plots : bool
        Whether to generate summary plots alongside CSV outputs.
    use_source_calibrator : bool
        If True, reuse the source calibrator from training / tuning.
    fit_calibrator : bool
        If True, fit a new calibrator on the inference dataset.
    calibrator_path : str or None
        Optional explicit calibrator path; may be None.
    """

    model_path: str
    dataset_key: str
    use_future_npz: bool
    manifest_path: Optional[str]

    inputs_npz: Optional[str]
    targets_npz: Optional[str]

    cov_target: float
    include_gwl: bool
    batch_size: int
    make_plots: bool

    use_source_calibrator: bool
    fit_calibrator: bool
    calibrator_path: Optional[str]


@dataclass
class InferencePlan:
    """
    Validated, backend-ready plan for an inference run.

    This is what the GUI callback will pass to InferenceThread.
    """

    model_path: str
    dataset_key: str
    use_stage1_future_npz: bool
    manifest_path: Optional[str]
    stage1_dir: Optional[str]

    inputs_npz: Optional[str]
    targets_npz: Optional[str]

    cov_target: float
    include_gwl: bool
    batch_size: int
    make_plots: bool

    use_source_calibrator: bool
    fit_calibrator: bool
    calibrator_path: Optional[str]


# ----------------------------------------------------------------------
# Controller
# ----------------------------------------------------------------------


class InferenceController(BaseWorkflowController):
    """
    Orchestrates planning and dry-run preview for inference.

    Responsibilities
    ----------------
    - Validate GUI state (model path, NPZ requirements).
    - Build a backend-friendly InferencePlan.
    - Provide:
        * dry_preview(...) to mirror _run_infer_dry_preview;
        * start_real_run(...) to mirror the planning / logging part
          of _on_infer_clicked, leaving thread creation to the GUI.

    The controller never touches Qt widgets directly; it only uses
    GUIHooks from BaseWorkflowController (log, status, progress, warn).
    """

    # ------------------------------------------------------------------
    # Planning
    # ------------------------------------------------------------------
    def plan_from_gui(
        self,
        state: InferenceGuiState,
        stage1_dir: Optional[str] = None,
    ) -> Optional[InferencePlan]:
        """
        Validate an InferenceGuiState and convert it into an
        InferencePlan.

        Returns
        -------
        InferencePlan or None
            None if validation fails (GUIHooks.warn is called).
        """
        # Basic validation: model is mandatory
        if not state.model_path:
            self.hooks.warn(
                "Model required",
                "Please select a trained/tuned .keras model first.",
            )
            # Keep progress consistent with original behaviour
            self.progress(0.0)
            return None

        # For custom NPZ without 'future', inputs_npz is required
        if (
            state.dataset_key == "custom"
            and not state.use_future_npz
            and not state.inputs_npz
        ):
            self.hooks.warn(
                "Inputs NPZ required",
                "For 'Custom NPZ', please select an inputs .npz file.",
            )
            self.progress(0.0)
            return None

        plan = InferencePlan(
            model_path=state.model_path,
            dataset_key=state.dataset_key or "test",
            use_stage1_future_npz=bool(state.use_future_npz),
            manifest_path=state.manifest_path or None,
            stage1_dir=stage1_dir,  # currently unused (was None in GUI)

            inputs_npz=state.inputs_npz or None,
            targets_npz=state.targets_npz or None,

            cov_target=float(state.cov_target),
            include_gwl=bool(state.include_gwl),
            batch_size=int(state.batch_size),
            make_plots=bool(state.make_plots),

            use_source_calibrator=bool(state.use_source_calibrator),
            fit_calibrator=bool(state.fit_calibrator),
            calibrator_path=state.calibrator_path or None,
        )
        return plan

    # ------------------------------------------------------------------
    # Dry-run preview
    # ------------------------------------------------------------------
    def dry_preview(
        self,
        state: InferenceGuiState,
        stage1_dir: Optional[str] = None,
    ) -> None:
        """
        Simulate what an inference run would do, without threads.

        Mirrors the behaviour of the original `_run_infer_dry_preview`:
        - validate settings;
        - log a detailed summary of what would happen;
        - drive the main progress bar 0 → 100 %;
        - no files are written, no models are loaded.
        """
        # Reset progress and status for dry mode
        self.progress(0.0)
        self.status("Dry-run / Inference: resolving inference settings…")

        # Build and validate plan
        plan = self.plan_from_gui(state, stage1_dir=stage1_dir)
        if plan is None:
            # Validation already warned and progress reset
            return

        # Intro message (same spirit as old helper)
        self.log(
            "[DRY] Inference preview – no model will be loaded, "
            "no files will be written."
        )

        # Mid-way progress: parsing + validation OK
        self.progress(0.4)

        # Summarise configuration in the log
        summary_lines = [
            "[DRY] Inference would run with:",
            f"  model_path       : {plan.model_path}",
            f"  dataset          : {plan.dataset_key}",
            f"  use_future_npz   : {plan.use_stage1_future_npz}",
            f"  manifest_path    : {plan.manifest_path or '<auto-discover>'}",
            f"  inputs_npz       : {plan.inputs_npz or '<Stage-1 default>'}",
            f"  targets_npz      : {plan.targets_npz or '<Stage-1 default>'}",
            f"  cov_target       : {plan.cov_target}",
            f"  include_gwl      : {plan.include_gwl}",
            f"  batch_size       : {plan.batch_size}",
            f"  make_plots       : {plan.make_plots}",
            f"  use_source_calib : {plan.use_source_calibrator}",
            f"  fit_calibrator   : {plan.fit_calibrator}",
            f"  calibrator_path  : {plan.calibrator_path or '<none>'}",
        ]
        for line in summary_lines:
            self.log(line)

        # Final progress
        self.progress(1.0)
        self.status(
            "[DRY] Inference preview complete – nothing was executed."
        )

    # ------------------------------------------------------------------
    # Real run
    # ------------------------------------------------------------------
    def start_real_run(
        self,
        state: InferenceGuiState,
        start_infer_cb: Callable[[InferencePlan], None],
        stage1_dir: Optional[str] = None,
    ) -> None:
        """
        Build an InferencePlan and, if valid, delegate to the GUI to
        start the actual InferenceThread.

        Parameters
        ----------
        state : InferenceGuiState
            Current GUI values.
        start_infer_cb : callable
            Callback provided by the GUI. It receives an InferencePlan
            and is responsible for:
                - creating InferenceThread;
                - wiring signals;
                - starting the thread;
                - managing buttons / timers.
        stage1_dir : str, optional
            Optional Stage-1 directory. The current GUI passes None;
            kept for future compatibility.
        """
        plan = self.plan_from_gui(state, stage1_dir=stage1_dir)
        if plan is None:
            return

        # Mirror the old logging / status behaviour
        self.log(
            f"Start inference: model={plan.model_path!r}, "
            f"dataset={plan.dataset_key!r}, "
            f"use_future={plan.use_stage1_future_npz}."
        )
        self.status("Stage-3: running inference.")
        self.progress(0.0)

        # Delegate thread creation to the GUI
        start_infer_cb(plan)
