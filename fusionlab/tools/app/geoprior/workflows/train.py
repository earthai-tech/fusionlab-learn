# geoprior/workflows/train.py
# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from ..services.stage1_service import (
    Stage1Decision,
    Stage1Service,
)
from .base import BaseWorkflowController, GUIHooks, RunEnv


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


class TrainController(BaseWorkflowController):
    def __init__(
        self,
        env: RunEnv,
        hooks: GUIHooks,
        stage1_svc: Stage1Service,
    ) -> None:
        super().__init__(env, hooks)
        self.stage1_svc = stage1_svc

    @staticmethod
    def _infer_city_from_path(path: Path) -> str:
        stem = path.stem.strip()
        if not stem:
            return "geoprior_city"
        return stem.replace(" ", "_")

    @staticmethod
    def _set_first_attr(
        obj: Any,
        names: tuple[str, ...],
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
        names: tuple[str, ...],
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

    def plan_from_gui(
        self,
        gui_state: TrainGuiState,
    ) -> Optional[TrainPlan]:
        cfg = self.env.geo_cfg
        store = getattr(self.env, "store", None)

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

        # --- Prefer store writes (single source of truth) ---
        if store is not None:
            try:
                store.set("city", city)
                store.set("dataset_path", csv_path_str)
            except Exception:
                pass

        # --- Backward-compat mirror into cfg ---
        self._set_first_attr(
            cfg,
            ("city", "CITY_NAME"),
            city,
        )
        self._set_first_attr(
            cfg,
            ("dataset_path", "TRAIN_CSV_PATH"),
            csv_path_str,
        )

        experiment_name = self._derive_experiment_name(
            cfg=cfg,
            requested=gui_state.experiment_name,
        )

        if store is not None:
            try:
                store.set(
                    "train.experiment_name",
                    experiment_name,
                )
            except Exception:
                pass

        self._set_first_attr(
            cfg,
            ("experiment_name", "EXPERIMENT_NAME"),
            experiment_name,
        )

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
        stage1_decision = self.stage1_svc.decide(
            city=city,
            clean_stage1_dir=clean_flag,
        )

        cfg_overrides = self.build_cfg_overrides()
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
