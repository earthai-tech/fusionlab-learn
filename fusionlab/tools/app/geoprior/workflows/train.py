# geoprior/workflows/train.py
# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

from ..services.stage1_service import (
    Stage1Decision,
    Stage1Service,
)
from .base import BaseWorkflowController, GUIHooks, RunEnv


__all__ = [
    "TrainGuiState",
    "TrainPlan",
    "TrainController",
]


@dataclass
class TrainGuiState:
    city_text: str
    csv_path: Optional[Path]
    experiment_name: Optional[str]


@dataclass
class TrainPlan:
    city: str
    csv_path: str
    cfg_overrides: Dict[str, Any]
    stage1_decision: Stage1Decision
    experiment_name: str


StartStage1Cb = Callable[[str, Dict[str, Any]], None]
StartTrainingCb = Callable[[str, Dict[str, Any]], None]


class TrainController(BaseWorkflowController):
    """
    Train workflow controller (v3.2 store-first).

    - Prefer store updates when available.
    - Mirror values into legacy GeoPriorConfig for v3.0.
    """

    def __init__(
        self,
        env: RunEnv,
        hooks: GUIHooks,
        stage1_svc: Stage1Service,
    ) -> None:
        super().__init__(env, hooks)
        self.stage1_svc = stage1_svc

    # -------------------------------------------------
    # Small helpers
    # -------------------------------------------------
    @staticmethod
    def _infer_city_from_path(path: Path) -> str:
        stem = path.stem.strip()
        if not stem:
            return "geoprior_city"
        return stem.replace(" ", "_")

    @staticmethod
    def _set_first_attr(
        obj: Any,
        names: Tuple[str, ...],
        value: Any,
    ) -> bool:
        for n in names:
            if hasattr(obj, n):
                try:
                    setattr(obj, n, value)
                    return True
                except Exception:
                    continue
        return False

    @staticmethod
    def _get_first_attr(
        obj: Any,
        names: Tuple[str, ...],
        default: Any = "",
    ) -> Any:
        for n in names:
            if hasattr(obj, n):
                try:
                    return getattr(obj, n)
                except Exception:
                    continue
        return default

    def _derive_experiment_name(
        self,
        cfg: Any,
        requested: Optional[str],
    ) -> str:
        if requested:
            name = requested.strip()
            if name:
                return name

        existing = self._get_first_attr(
            cfg,
            ("experiment_name", "EXPERIMENT_NAME"),
            "",
        )
        if existing:
            return str(existing)

        city = self._get_first_attr(
            cfg,
            ("city", "CITY_NAME"),
            "",
        )
        if city:
            return str(city)

        return "geoprior_run"

    # -------------------------------------------------
    # Stage-1 decision normalization (v3.2-first)
    # -------------------------------------------------
    def _decision_manifest_path(self, dec: Any) -> Optional[str]:
        """
        Return Stage-1 manifest path from a decision.

        v3.2: prefers `manifest_hint`
        legacy: checks common historical attribute names
        """
        if dec is None:
            return None

        # v3.2 contract
        try:
            v = getattr(dec, "manifest_hint", None)
            if v:
                return str(v)
        except Exception:
            pass

        # legacy fallbacks
        for a in ("manifest_path", "manifest", "path", "stage1_manifest"):
            if hasattr(dec, a):
                try:
                    v = getattr(dec, a)
                    if v:
                        return str(v)
                except Exception:
                    continue

        return None

    def _decision_mode(self, dec: Any) -> str:
        """
        Return: "cancel" | "reuse" | "build"

        v3.2 contract:
        - cancelled -> cancel
        - not need_stage1 and manifest_hint -> reuse
        - else -> build

        legacy fallback: allow old action/kind/mode semantics.
        """
        if dec is None:
            return "cancel"

        if bool(getattr(dec, "cancelled", False)):
            return "cancel"

        # v3.2 fields (preferred)
        need_stage1 = getattr(dec, "need_stage1", None)
        mp = self._decision_manifest_path(dec)

        if need_stage1 is not None:
            if (not bool(need_stage1)) and mp:
                return "reuse"
            return "build"

        # legacy fallback (only if need_stage1 absent)
        act = None
        for a in ("action", "kind", "mode", "name", "value"):
            if hasattr(dec, a):
                try:
                    act = getattr(dec, a)
                    if act is not None:
                        break
                except Exception:
                    continue

        s = str(act if act is not None else dec).lower()

        if any(k in s for k in ("cancel", "abort", "stop")):
            return "cancel"
        if any(k in s for k in ("reuse", "existing", "use")):
            return "reuse"
        if any(k in s for k in (
                "build", "rebuild", "run", "stage1", 
                "preprocess", "scratch")):
            return "build"

        return "reuse" if mp else "build"

    # -------------------------------------------------
    # Planning
    # -------------------------------------------------
    def plan_from_gui(
        self,
        gui_state: TrainGuiState,
    ) -> Optional[TrainPlan]:
        cfg = self.resolve_cfg()
        st = getattr(self.env, "store", None)

        if gui_state.csv_path is None:
            self.hooks.warn(
                "No training dataset",
                "Please choose a training data file first.",
            )
            self.status("Train: aborted (no training data).")
            self.progress(0.0)
            return None

        csv_path = gui_state.csv_path
        csv_str = str(csv_path)

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

        exp_name = self._derive_experiment_name(
            cfg=cfg,
            requested=gui_state.experiment_name,
        )

        # --- Prefer store writes (v3.2) ---
        if st is not None:
            try:
                with st.batch():
                    st.set("city", city)
                    st.set("dataset_path", csv_str)
                    st.set("experiment_name", exp_name)
            except Exception:
                pass

        # --- Mirror into cfg (v3.0 compat) ---
        self._set_first_attr(cfg, ("city",), city)
        self._set_first_attr(cfg, ("CITY_NAME",), city)

        # dataset_path is Path in v3.2 store
        self._set_first_attr(cfg, ("dataset_path",), csv_path)
        self._set_first_attr(
            cfg,
            ("TRAIN_CSV_PATH",),
            csv_str,
        )

        self._set_first_attr(
            cfg,
            ("experiment_name",),
            exp_name,
        )
        self._set_first_attr(
            cfg,
            ("EXPERIMENT_NAME",),
            exp_name,
        )

        # Validate if supported
        try:
            if hasattr(cfg, "ensure_valid"):
                cfg.ensure_valid()
        except Exception as exc:
            self.hooks.error(
                "Invalid configuration",
                "The training configuration is not valid:\n\n"
                f"{exc}",
            )
            self.status("Train: aborted (invalid configuration).")
            self.progress(0.0)
            return None

        clean_flag = bool(
            self._get_first_attr(
                cfg,
                ("clean_stage1_dir",),
                False,
            )
        )

        stage1_dec = self.stage1_svc.decide(
            city=city,
            clean_stage1_dir=clean_flag,
        )

        cfg_ov = self.build_cfg_overrides()

        cfg_ov.setdefault(
            "BASE_OUTPUT_DIR",
            str(self.env.gui_runs_root),
        )

        # Ensure legacy consumers have these
        cfg_ov.setdefault("TRAIN_CSV_PATH", csv_str)
        cfg_ov.setdefault("CITY_NAME", city)
        cfg_ov.setdefault("city", city)
        cfg_ov.setdefault("EXPERIMENT_NAME", exp_name)

        return TrainPlan(
            city=city,
            csv_path=csv_str,
            cfg_overrides=cfg_ov,
            stage1_decision=stage1_dec,
            experiment_name=exp_name,
        )

    # -------------------------------------------------
    # Dry preview (app.py)
    # -------------------------------------------------
    def dry_preview(self, gui_state: TrainGuiState) -> None:
        plan = self.plan_from_gui(gui_state)
        if plan is None:
            return

        mode = self._decision_mode(plan.stage1_decision)
        mp = self._decision_manifest_path(plan.stage1_decision)

        self.log("DRY RUN — Train plan")
        self.log(f"  city: {plan.city}")
        self.log(f"  csv:  {plan.csv_path}")
        self.log(f"  exp:  {plan.experiment_name}")
        self.log(f"  stage1: {mode}")
        if mp:
            self.log(f"  manifest: {mp}")

        self.status("Dry-run: train plan ready.")
        self.progress(0.0)

            
    # -------------------------------------------------
    # Real run (app.py)
    # -------------------------------------------------
    def start_real_run(
        self,
        gui_state: TrainGuiState,
        start_stage1_cb: StartStage1Cb,
        start_training_cb: StartTrainingCb,
    ) -> None:
        """
        Execute the train workflow using app-provided callbacks.
    
        Stage1Decision contract (exact fields):
        - dec.cancelled
        - dec.need_stage1
        - dec.manifest_hint
        """
        if getattr(self.env, "dry_mode", False):
            self.dry_preview(gui_state)
            return
    
        plan = self.plan_from_gui(gui_state)
        if plan is None:
            return
    
        dec = plan.stage1_decision
    
        try:
            if getattr(dec, "cancelled", False):
                self.status("Train: cancelled.")
                self.progress(0.0)
                return
    
            # Reuse Stage-1 if allowed and manifest is available
            if (not getattr(dec, "need_stage1", True)) and getattr(
                dec, "manifest_hint", None
            ):
                self.status("Train: reusing existing Stage-1.")
                start_training_cb(dec.manifest_hint, plan.cfg_overrides)
                return
    
            # Otherwise build Stage-1 from scratch, then GUI will continue
            self.status("Train: starting Stage-1…")
            start_stage1_cb(plan.city, plan.cfg_overrides)
    
        except Exception as exc:
            self.hooks.error(
                "Train failed",
                "Could not start the training workflow:\n\n"
                f"{exc}",
            )
            self.status("Train: failed to start.")
            self.progress(0.0)

