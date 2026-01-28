# geoprior/workflows/preprocess.py
# -*- coding: utf-8 -*-
# License: BSD-3-Clause
#
# Preprocess (Stage-1) workflow controller for the GeoPrior GUI.
#
# v3.2 goals
# ----------
# - Store-first: read/write through env.store when available.
# - Back-compat: fall back to env.resolve_cfg() attributes.
# - Qt-free: no PyQt imports; GUI effects via GUIHooks + callbacks.

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from .base import BaseWorkflowController

from ..services.stage1_service import Stage1Service

@dataclass
class PreprocessGuiState:
    """
    Minimal snapshot of the Preprocess tab (Qt-free).

    Notes
    -----
    - city and dataset_path must be resolvable from GUI or store/cfg.
    - results_root can be resolved from cfg or defaults to gui_runs_root.
    """

    city: str
    dataset_path: Optional[str]
    results_root: Optional[str]

    # stage-1 options (optional overrides)
    clean_stage1_dir: Optional[bool] = None
    build_future_npz: Optional[bool] = None
    auto_reuse_if_match: Optional[bool] = None
    force_rebuild_if_mismatch: Optional[bool] = None


@dataclass
class PreprocessPlan:
    """
    Validated plan passed to the GUI callback / Stage1Thread.
    """

    city: str
    dataset_path: str
    results_root: str

    cfg_overrides: Dict[str, Any]
    config_overwrite: Dict[str, Any]

    clean_run_dir: bool
    build_future_npz: bool

    # optional: if controller decides reuse
    reuse_manifest: Optional[str] = None


StartStage1Cb = Callable[[PreprocessPlan], None]
ReuseStage1Cb = Callable[[str], None]

class PreprocessController(BaseWorkflowController):
    """
    Plan + dry-run logic for preprocess (Stage-1).

    Behaviour
    ---------
    - Read missing values from store first (if provided).
    - Mirror key values back into store (optional).
    - Reuse decision can be delegated to Stage1Service if provided.
    """

    def __init__(
        self,
        *,
        env: Any,
        hooks: Any,
        stage1_svc: Optional[Stage1Service] = None,
    ) -> None:
        super().__init__(env=env, hooks=hooks)
        self.stage1_svc = stage1_svc

    # -----------------------------
    # store helpers (best-effort)
    # -----------------------------
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

    # -----------------------------
    # planning
    # -----------------------------
    def plan_from_gui(
        self,
        state: PreprocessGuiState,
    ) -> Optional[PreprocessPlan]:
        cfg = self.resolve_cfg()

        city = str(state.city or "").strip()
        if not city:
            city = str(getattr(cfg, "city", "") or "").strip()
        if not city:
            self.hooks.warn(
                "City required",
                "Select a city before running Stage-1.",
            )
            self.progress(0.0)
            return None

        ds = (state.dataset_path or "").strip() or None
        if ds is None:
            ds = str(getattr(cfg, "dataset_path", "") or "").strip()
            ds = ds or None
        if ds is None:
            self.hooks.warn(
                "Dataset required",
                "Select a CSV dataset for Stage-1.",
            )
            self.progress(0.0)
            return None

        rr = (state.results_root or "").strip() or None
        if rr is None:
            rr = str(getattr(cfg, "results_root", "") or "").strip()
            rr = rr or None

        if rr is None:
            root = Path(getattr(self.env, "gui_runs_root", "."))
            rr = str((root / city).resolve())

        # stage-1 options
        clean_dir = state.clean_stage1_dir
        if clean_dir is None:
            clean_dir = bool(getattr(cfg, "clean_stage1_dir", False))

        build_future = state.build_future_npz
        if build_future is None:
            build_future = bool(getattr(cfg, "build_future_npz", True))

        auto_reuse = state.auto_reuse_if_match
        if auto_reuse is None:
            auto_reuse = bool(
                getattr(cfg, "stage1_auto_reuse_if_match", True)
            )

        force_rebuild = state.force_rebuild_if_mismatch
        if force_rebuild is None:
            force_rebuild = bool(
                getattr(cfg, "stage1_force_rebuild_if_mismatch", True)
            )

        # cfg_overrides (store-first)
        overrides = self.build_cfg_overrides()

        # keep same convention as train controller
        overrides["BASE_OUTPUT_DIR"] = str(
            getattr(self.env, "gui_runs_root", ".")
        )

        plan = PreprocessPlan(
            city=city,
            dataset_path=ds,
            results_root=rr,
            cfg_overrides=overrides,
            config_overwrite={},
            clean_run_dir=bool(clean_dir),
            build_future_npz=bool(build_future),
            reuse_manifest=None,
        )

        # mirror convenience values into store
        self._store_set("prep.city", plan.city)
        self._store_set("prep.dataset_path", plan.dataset_path)
        self._store_set("prep.results_root", plan.results_root)
        self._store_set("prep.build_future_npz", plan.build_future_npz)
        self._store_set("prep.clean_stage1_dir", plan.clean_run_dir)
        self._store_set("prep.auto_reuse", bool(auto_reuse))
        self._store_set("prep.force_rebuild", bool(force_rebuild))

        # reuse decision (optional)
        if self.stage1_svc is not None:
            dec = self.stage1_svc.decide(
                city=plan.city,
                clean_stage1_dir=plan.clean_run_dir,
            )
            manifest = getattr(dec, "manifest_hint", None)
            need = bool(getattr(dec, "need_stage1", True))
            cancelled = bool(getattr(dec, "cancelled", False))

            if cancelled:
                self.hooks.warn(
                    "Preprocess cancelled",
                    "Stage-1 decision returned cancelled=True.",
                )
                return None

            # respect force_rebuild and auto_reuse
            if (not need) and manifest and (not force_rebuild):
                if auto_reuse and (not plan.clean_run_dir):
                    plan.reuse_manifest = str(manifest)

        return plan

    # -----------------------------
    # dry preview
    # -----------------------------
    def dry_preview(self, state: PreprocessGuiState) -> None:
        self.progress(0.0)
        self.status("Dry-run / Preprocess: resolving settings…")

        plan = self.plan_from_gui(state)
        if plan is None:
            return

        self.log(
            "[DRY] Preprocess preview – no Stage-1 job will run."
        )
        self.progress(0.5)

        reuse = plan.reuse_manifest or "<none>"
        lines = [
            "[DRY] Preprocess plan:",
            f"  city            : {plan.city}",
            f"  dataset_path    : {plan.dataset_path}",
            f"  results_root    : {plan.results_root}",
            f"  clean_run_dir   : {plan.clean_run_dir}",
            f"  build_future_npz: {plan.build_future_npz}",
            f"  reuse_manifest  : {reuse}",
        ]
        for line in lines:
            self.log(line)

        self.progress(1.0)
        self.status(
            "[DRY] Preprocess preview complete – nothing executed."
        )

    # -----------------------------
    # real run
    # -----------------------------
    def start_real_run(
        self,
        state: PreprocessGuiState,
        start_stage1_cb: StartStage1Cb,
        *,
        reuse_stage1_cb: Optional[ReuseStage1Cb] = None,
    ) -> None:
        if getattr(self.env, "dry_mode", False):
            self.dry_preview(state)
            return

        plan = self.plan_from_gui(state)
        if plan is None:
            return

        if plan.reuse_manifest and reuse_stage1_cb is not None:
            self.log(
                "Preprocess: reuse existing Stage-1 "
                f"({plan.reuse_manifest})."
            )
            self.status("Stage-1: reusing existing artifacts.")
            self.progress(1.0)
            reuse_stage1_cb(plan.reuse_manifest)
            return

        self.log(
            "Start preprocess (Stage-1): "
            f"city={plan.city!r}, "
            f"clean={plan.clean_run_dir}, "
            f"future_npz={plan.build_future_npz}."
        )
        self.status("Stage-1: running preprocess.")
        self.progress(0.0)

        start_stage1_cb(plan)
