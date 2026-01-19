# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>
#
# Transferability workflow controller.
#
# Encapsulates planning and dry-run logic for the cross-city transfer
# matrix (XferMatrixThread), so the Qt GUI only needs to:
#
#   * collect widget values into a TransferGuiState;
#   * delegate validation + logging to TransferController;
#   * provide a small callback that knows how to actually start the
#     XferMatrixThread from a TransferPlan.
#
# This mirrors the structure of TrainController / TuneController /
# InferenceController.

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Callable

from .base import BaseWorkflowController, RunEnv, GUIHooks


# ----------------------------------------------------------------------
# Data structures
# ----------------------------------------------------------------------

@dataclass
class TransferGuiState:
    """
    Snapshot of the Transferability tab.

    Plain Python types only (Qt-free).
    """

    city_a: str
    city_b: str
    results_root: Optional[str]

    splits: List[str]
    calib_modes: List[str]

    rescale_to_source: bool
    batch_size: int
    quantiles_override: Optional[Sequence[float]]

    write_json: bool
    write_csv: bool

    # ---- v3.2 extras (all optional / defaulted) ----
    rescale_modes: Optional[List[str]] = None
    strategies: Optional[List[str]] = None

    prefer_tuned: bool = True
    align_policy: str = "align_by_name_pad"

    allow_reorder_dynamic: Optional[bool] = None
    allow_reorder_future: Optional[bool] = None

    warm_split: Optional[str] = None
    warm_samples: Optional[int] = None
    warm_frac: Optional[float] = None
    warm_epochs: Optional[int] = None
    warm_lr: Optional[float] = None
    warm_seed: Optional[int] = None


@dataclass
class TransferPlan:
    """
    Fully validated plan passed to XferMatrixThread.
    """

    city_a: str
    city_b: str
    results_root: Path

    splits: List[str]
    calib_modes: List[str]

    rescale_to_source: bool
    batch_size: int
    quantiles_override: Optional[List[float]]

    write_json: bool
    write_csv: bool

    # ---- v3.2 extras ----
    rescale_modes: Optional[List[str]] = None
    strategies: Optional[List[str]] = None

    prefer_tuned: bool = True
    align_policy: str = "align_by_name_pad"

    allow_reorder_dynamic: Optional[bool] = None
    allow_reorder_future: Optional[bool] = None

    warm_split: Optional[str] = None
    warm_samples: Optional[int] = None
    warm_frac: Optional[float] = None
    warm_epochs: Optional[int] = None
    warm_lr: Optional[float] = None
    warm_seed: Optional[int] = None


# ----------------------------------------------------------------------
# Controller
# ----------------------------------------------------------------------


class TransferController(BaseWorkflowController):
    """
    Orchestrates planning for the cross-city transfer matrix.

    Responsibilities
    ----------------
    * validate GUI state (cities, splits, calibration modes);
    * provide a dry-run preview (logs + progress, no threads);
    * create a TransferPlan and hand it to a callback for real runs.
    """

    def __init__(self, env: RunEnv, hooks: GUIHooks):
        super().__init__(env, hooks)

    # ------------------------------------------------------------------
    # Internal helper: validate & build plan
    # ------------------------------------------------------------------
    def _build_plan(
        self,
        state: TransferGuiState,
        *,
        for_dry_run: bool = False,
    ) -> Optional[TransferPlan]:
        """
        Turn a TransferGuiState into a TransferPlan.

        Parameters
        ----------
        state : TransferGuiState
            Current GUI snapshot.
        for_dry_run : bool, default False
            If True, allow the Nansha/Zhongshan defaults when cities
            are left blank (to mirror your old _run_xfer_dry_preview
            behaviour). For real runs, both cities must be provided.

        Returns
        -------
        TransferPlan or None
            None indicates validation failed (warnings already shown).
        """
        # --- Resolve results_root ------------------------------------
        root_txt = state.results_root or ""
        if root_txt.strip():
            results_root = Path(root_txt.strip())
        else:
            results_root = Path(self.env.gui_runs_root)

        # --- Cities ---------------------------------------------------
        city_a = (state.city_a or "").strip()
        city_b = (state.city_b or "").strip()

        if for_dry_run:
            # Keep previous behaviour: default to Nansha/Zhongshan
            # *only* for the dry preview.
            if not city_a:
                city_a = "nansha"
            if not city_b:
                city_b = "zhongshan"

        if not city_a or not city_b:
            self.hooks.warn(
                "Cities required",
                "Please fill both City A and City B.",
            )
            self.progress(0.0)
            return None

        # --- Splits ---------------------------------------------------
        splits = [s for s in (state.splits or []) if s]
        if not splits:
            self.hooks.warn(
                "Splits required",
                "Please select at least one split.",
            )
            self.progress(0.0)
            return None

        # --- Calibration modes ---------------------------------------
        calib_modes = [m for m in (state.calib_modes or []) if m]
        if not calib_modes:
            self.hooks.warn(
                "Calibration modes required",
                "Please select at least one calibration mode.",
            )
            self.progress(0.0)
            return None

        q_override: Optional[List[float]] = None
        if state.quantiles_override is not None:
            q_override = [float(q) for q in state.quantiles_override]

        rescale_modes = (
            [m for m in (state.rescale_modes or []) if m]
            if state.rescale_modes is not None
            else None
        )

        strategies = (
            [s for s in (state.strategies or []) if s]
            if state.strategies is not None
            else None
        )

        return TransferPlan(
            city_a=city_a,
            city_b=city_b,
            results_root=results_root,
            splits=splits,
            calib_modes=calib_modes,
            rescale_to_source=bool(state.rescale_to_source),
            batch_size=int(state.batch_size),
            quantiles_override=q_override,
            write_json=bool(state.write_json),
            write_csv=bool(state.write_csv),
            rescale_modes=rescale_modes,
            strategies=strategies,
            prefer_tuned=bool(state.prefer_tuned),
            align_policy=str(state.align_policy or "align_by_name_pad"),
            allow_reorder_dynamic=state.allow_reorder_dynamic,
            allow_reorder_future=state.allow_reorder_future,
            warm_split=state.warm_split,
            warm_samples=state.warm_samples,
            warm_frac=state.warm_frac,
            warm_epochs=state.warm_epochs,
            warm_lr=state.warm_lr,
            warm_seed=state.warm_seed,
        )


    # ------------------------------------------------------------------
    # Public API: dry preview
    # ------------------------------------------------------------------
    def dry_preview(self, state: TransferGuiState) -> None:
        """
        Simulate what a transfer run would do, without starting threads.

        Mirrors the old `_run_xfer_dry_preview` behaviour:

        - allows city defaults (Nansha/Zhongshan) if fields are blank;
        - validates splits and calibration modes;
        - logs a textual plan and drives the main progress bar.
        """
        # start from 0 %
        self.progress(0.0)
        self.status(
            "[DRY] Transferability preview – no models will be loaded, "
            "no transfer matrix will be computed."
        )

        plan = self._build_plan(state, for_dry_run=True)
        if plan is None:
            # Validation already handled via hooks.warn
            return

        # Mid-way: everything parsed
        self.progress(0.5)

        q_display = (
            plan.quantiles_override if plan.quantiles_override is not None
            else "<from model>"
        )

        lines = [
            "[DRY] Transfer matrix would run with:",
            f"  city_a       : {plan.city_a}",
            f"  city_b       : {plan.city_b}",
            f"  results_root : {plan.results_root}",
            f"  splits       : {plan.splits}",
            f"  calib_modes  : {plan.calib_modes}",
            f"  rescale      : {plan.rescale_to_source}",
            f"  batch_size   : {plan.batch_size}",
            f"  quantiles    : {q_display}",
            f"  write_json   : {plan.write_json}",
            f"  write_csv    : {plan.write_csv}",
            f"  strategies   : {plan.strategies or '<default>'}",
            f"  rescale_modes: {plan.rescale_modes or '<auto>'}",
            f"  prefer_tuned : {plan.prefer_tuned}",
            f"  align_policy : {plan.align_policy}",
            f"  allow_re_dyn : {plan.allow_reorder_dynamic}",
            f"  allow_re_fut : {plan.allow_reorder_future}",
            f"  warm_split   : {plan.warm_split}",
            f"  warm_samples : {plan.warm_samples}",
            f"  warm_frac    : {plan.warm_frac}",
            f"  warm_epochs  : {plan.warm_epochs}",
            f"  warm_lr      : {plan.warm_lr}",
            f"  warm_seed    : {plan.warm_seed}",

        ]
        for line in lines:
            self.log(line)

        # Final progress + status
        self.progress(1.0)
        self.status(
            "[DRY] Transferability preview complete – nothing was executed."
        )

    # ------------------------------------------------------------------
    # Public API: real run
    # ------------------------------------------------------------------
    def start_real_run(
        self,
        state: TransferGuiState,
        start_xfer_cb: Callable[[TransferPlan], None],
    ) -> None:
        """
        Validate GUI state and, if successful, start a real transfer run.

        Parameters
        ----------
        state : TransferGuiState
            Snapshot of the Transferability tab.
        start_xfer_cb : callable
            Callback ``f(plan: TransferPlan) -> None`` that knows how to
            instantiate and start XferMatrixThread from the plan
            (this is provided by the GeoPriorForecaster GUI).
        """
        plan = self._build_plan(state, for_dry_run=False)
        if plan is None:
            # Validation already handled
            return

        # Mirror your previous logging
        self.log(
            "Start transfer matrix: "
            f"{plan.city_a!r} ↔ {plan.city_b!r}; "
            f"splits={plan.splits}, "
            f"calib={plan.calib_modes}, "
            f"strat={plan.strategies or 'default'}, "
            f"rescale={plan.rescale_to_source}, "
            f"modes={plan.rescale_modes or 'auto'}."
        )
        self.status(
            f"XFER: running transfer matrix for "
            f"{plan.city_a} and {plan.city_b}."
        )
        self.progress(0.0)

        # Hand off to the GUI callback which will:
        #   * start the run timer;
        #   * create XferMatrixThread;
        #   * wire signals & buttons;
        #   * th.start().
        start_xfer_cb(plan)
