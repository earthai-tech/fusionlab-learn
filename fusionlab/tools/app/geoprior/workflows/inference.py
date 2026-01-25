# geoprior/workflows/inference.py
# -*- coding: utf-8 -*-
# License: BSD-3-Clause
#
# Inference workflow controller for the GeoPrior GUI.
#
# v3.2 goals
# ----------
# - Store-first: read/write through env.store when available.
# - Back-compat: fall back to env.resolve_cfg() attributes.
# - Qt-free: the controller never imports PyQt; GUI side-effects
#   go through GUIHooks + a GUI callback that starts InferenceThread.
#
# Compatibility
# -------------
# Mirrors the intent of the legacy GUI handlers:
# - _on_infer_clicked (planning + logging)
# - _run_infer_dry_preview (dry-run summary + progress)

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

from .base import BaseWorkflowController


# ---------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------


@dataclass
class InferenceGuiState:
    """
    Minimal snapshot of the Inference tab (Qt-free).

    Notes
    -----
    - model_path is required for all runs.
    - dataset_key is typically one of {"train","val","test","custom"}.
    - for dataset_key="custom", inputs_npz is required when not using
      the Stage-1 future npz.
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
    Validated plan passed to the GUI callback / InferenceThread.
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


StartInferCb = Callable[[InferencePlan], None]


# ---------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------


class InferenceController(BaseWorkflowController):
    """
    Plan + dry-run logic for inference.

    v3.2 behaviour
    -------------
    - Read missing values from store first (if provided).
    - Mirror key values back into store so the UI is persistent
      across sessions.
    - Do not assume schema keys exist: "infer.*" keys can live in the
      store's GUI-only extras.
    """

    # -----------------------------------------------------------------
    # Store helpers (best-effort)
    # -----------------------------------------------------------------
    def _store_get(self, key: str, default: Any = None) -> Any:
        st = getattr(self.env, "store", None)
        if st is not None and hasattr(st, "get"):
            try:
                return st.get(key, default)
            except Exception:
                return default
        return default

    def _store_set(self, key: str, value: Any) -> None:
        st = getattr(self.env, "store", None)
        if st is None or not hasattr(st, "set"):
            return
        try:
            st.set(key, value)
        except Exception:
            pass

    # -----------------------------------------------------------------
    # Planning
    # -----------------------------------------------------------------
    def plan_from_gui(
        self,
        state: InferenceGuiState,
        *,
        stage1_dir: Optional[str] = None,
    ) -> Optional[InferencePlan]:
        """
        Validate GUI state and convert it into an InferencePlan.

        Precedence (v3.2)
        -----------------
        - model_path:
            GUI -> store("infer.model_path") (optional convenience)
        - manifest_path:
            GUI -> store("infer.manifest_path") -> None (auto-discover)
        - calibrator_path:
            GUI -> store("infer.calibrator_path") -> None

        Returns
        -------
        InferencePlan or None
            None indicates validation failed (hooks.warn called).
        """
        # -----------------------------
        # model path (required)
        # -----------------------------
        model_path = str(state.model_path or "").strip()
        if not model_path:
            model_path = str(
                self._store_get("infer.model_path", "")
            ).strip()

        if not model_path:
            self.hooks.warn(
                "Model required",
                "Please select a trained/tuned .keras model first.",
            )
            self.progress(0.0)
            return None

        # -----------------------------
        # dataset selection
        # -----------------------------
        dataset_key = str(state.dataset_key or "").strip().lower()
        if not dataset_key:
            dataset_key = str(
                self._store_get("infer.dataset_key", "test")
            ).strip().lower()
        if dataset_key not in ("train", "val", "test", "custom"):
            # Be forgiving: keep unknown keys but default to test
            dataset_key = "test"

        use_future = bool(state.use_future_npz)

        # -----------------------------
        # custom npz validation
        # -----------------------------
        inputs_npz = (state.inputs_npz or None)
        targets_npz = (state.targets_npz or None)

        if dataset_key == "custom" and (not use_future):
            if not inputs_npz:
                # store fallback (optional)
                inputs_npz = self._store_get(
                    "infer.inputs_npz", None
                )
                inputs_npz = str(inputs_npz).strip() if inputs_npz else None

            if not inputs_npz:
                self.hooks.warn(
                    "Inputs NPZ required",
                    "For 'Custom NPZ', please select an inputs .npz file.",
                )
                self.progress(0.0)
                return None

        # -----------------------------
        # manifest / calibrator paths
        # -----------------------------
        manifest_path = (state.manifest_path or "").strip() or None
        if manifest_path is None:
            mp = self._store_get("infer.manifest_path", None)
            manifest_path = str(mp).strip() if mp else None

        calibrator_path = (state.calibrator_path or "").strip() or None
        if calibrator_path is None:
            cp = self._store_get("infer.calibrator_path", None)
            calibrator_path = str(cp).strip() if cp else None

        # -----------------------------
        # numeric + flags
        # -----------------------------
        try:
            cov_target = float(state.cov_target)
        except Exception:
            cov_target = float(self._store_get("infer.cov_target", 0.8) or 0.8)

        try:
            batch_size = int(state.batch_size)
        except Exception:
            batch_size = int(self._store_get("infer.batch_size", 256) or 256)

        include_gwl = bool(state.include_gwl)
        make_plots = bool(state.make_plots)
        use_source_cal = bool(state.use_source_calibrator)
        fit_cal = bool(state.fit_calibrator)

        plan = InferencePlan(
            model_path=model_path,
            dataset_key=dataset_key,
            use_stage1_future_npz=use_future,
            manifest_path=manifest_path,
            stage1_dir=stage1_dir,
            inputs_npz=(inputs_npz or None),
            targets_npz=(targets_npz or None),
            cov_target=cov_target,
            include_gwl=include_gwl,
            batch_size=batch_size,
            make_plots=make_plots,
            use_source_calibrator=use_source_cal,
            fit_calibrator=fit_cal,
            calibrator_path=calibrator_path,
        )

        # -----------------------------
        # mirror into store (v3.2)
        # -----------------------------
        self._store_set("infer.model_path", plan.model_path)
        self._store_set("infer.dataset_key", plan.dataset_key)
        self._store_set("infer.use_future_npz", plan.use_stage1_future_npz)
        self._store_set("infer.manifest_path", plan.manifest_path)
        self._store_set("infer.inputs_npz", plan.inputs_npz)
        self._store_set("infer.targets_npz", plan.targets_npz)
        self._store_set("infer.cov_target", plan.cov_target)
        self._store_set("infer.include_gwl", plan.include_gwl)
        self._store_set("infer.batch_size", plan.batch_size)
        self._store_set("infer.make_plots", plan.make_plots)
        self._store_set(
            "infer.use_source_calibrator",
            plan.use_source_calibrator,
        )
        self._store_set("infer.fit_calibrator", plan.fit_calibrator)
        self._store_set("infer.calibrator_path", plan.calibrator_path)

        return plan

    # -----------------------------------------------------------------
    # Dry preview
    # -----------------------------------------------------------------
    def dry_preview(
        self,
        state: InferenceGuiState,
        *,
        stage1_dir: Optional[str] = None,
    ) -> None:
        """
        Dry-run preview: validate + log what would happen.

        No model load, no IO, no threads.
        """
        self.progress(0.0)
        self.status("Dry-run / Inference: resolving settings…")

        plan = self.plan_from_gui(state, stage1_dir=stage1_dir)
        if plan is None:
            return

        self.log(
            "[DRY] Inference preview – no model will be loaded, "
            "no files will be written."
        )
        self.progress(0.4)

        manifest = plan.manifest_path or "<auto-discover>"
        inputs_npz = plan.inputs_npz or "<Stage-1 default>"
        targets_npz = plan.targets_npz or "<Stage-1 default>"
        calib = plan.calibrator_path or "<none>"

        lines = [
            "[DRY] Inference plan:",
            f"  model_path        : {plan.model_path}",
            f"  dataset           : {plan.dataset_key}",
            f"  use_future_npz    : {plan.use_stage1_future_npz}",
            f"  manifest_path     : {manifest}",
            f"  inputs_npz        : {inputs_npz}",
            f"  targets_npz       : {targets_npz}",
            f"  cov_target        : {plan.cov_target}",
            f"  include_gwl       : {plan.include_gwl}",
            f"  batch_size        : {plan.batch_size}",
            f"  make_plots        : {plan.make_plots}",
            f"  use_source_calib  : {plan.use_source_calibrator}",
            f"  fit_calibrator    : {plan.fit_calibrator}",
            f"  calibrator_path   : {calib}",
        ]
        for line in lines:
            self.log(line)

        self.progress(1.0)
        self.status(
            "[DRY] Inference preview complete – nothing was executed."
        )

    # -----------------------------------------------------------------
    # Real run
    # -----------------------------------------------------------------
    def start_real_run(
        self,
        state: InferenceGuiState,
        start_infer_cb: StartInferCb,
        *,
        stage1_dir: Optional[str] = None,
    ) -> None:
        """
        Real run entry point.

        The GUI callback is responsible for creating/wiring
        InferenceThread(plan) and starting it.
        """
        if getattr(self.env, "dry_mode", False):
            self.dry_preview(state, stage1_dir=stage1_dir)
            return

        plan = self.plan_from_gui(state, stage1_dir=stage1_dir)
        if plan is None:
            return

        self.log(
            "Start inference: "
            f"model={plan.model_path!r}, "
            f"dataset={plan.dataset_key!r}, "
            f"use_future={plan.use_stage1_future_npz}."
        )
        self.status("Stage-3: running inference.")
        self.progress(0.0)

        start_infer_cb(plan)
