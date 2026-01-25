# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>
#
# geoprior/workflows/tune.py
#
# Stage-2 hyperparameter tuning workflow controller.
#
# Design goals (v3.2)
# -------------------
# - Store-first: GeoConfigStore is the single source of truth when
#   available (env.store). It emits signals and tracks overrides.
# - Backward compatible: if no store is provided, fall back to the
#   legacy GeoPriorConfig instance (env.geo_cfg).
# - Qt-free: this controller must not import PyQt. All UI actions are
#   routed through GUIHooks and a GUI-provided callback that starts
#   the actual Qt thread (TuningThread).

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from ..config import default_tuner_search_space
from .base import BaseWorkflowController, RunEnv, GUIHooks


# ---------------------------------------------------------------------
# Data objects exchanged between GUI and workflow
# ---------------------------------------------------------------------


@dataclass
class TuneGuiState:
    """
    Minimal snapshot of what the GUI provides for tuning.

    Notes
    -----
    - Most fields can be None because the GUI might resolve them from:
      * a Stage-1 manifest picker dialog
      * an auto-discovery action
      * cached config/store state
    - The controller is responsible for normalizing/filling defaults.
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
    Fully validated tuning plan used by the backend / threads.

    The GUI callback receives this object and is responsible for:
    - creating the tuning thread/job
    - wiring signals
    - managing UI enabling/disabling/timers
    """

    city: str
    manifest_path: Optional[str]
    stage1_root: Optional[str]
    cfg_overrides: Dict[str, Any]
    max_trials: int


# ---------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------


class TuneController(BaseWorkflowController):
    """
    Encapsulate Stage-2 tuning workflow (real runs + dry preview).

    Key idea
    --------
    `plan_from_gui()` is the only place where we:
    - validate the GUI state,
    - resolve missing values with store/cfg fallbacks,
    - build the NAT-style cfg_overrides map that the backend expects.
    """

    def __init__(self, env: RunEnv, hooks: GUIHooks) -> None:
        super().__init__(env, hooks)

    # -----------------------------------------------------------------
    # Internals
    # -----------------------------------------------------------------
    def _store_get(self, key: str, default: Any = None) -> Any:
        """
        Best-effort store read.

        We intentionally keep this very defensive:
        - store may not exist
        - store may not implement .get()
        - store may throw (shouldn't, but avoid crashing the workflow)
        """
        st = getattr(self.env, "store", None)
        if st is not None and hasattr(st, "get"):
            try:
                return st.get(key, default)
            except Exception:
                return default
        return default

    def _store_set(self, key: str, value: Any) -> None:
        """
        Best-effort store write (v3.2).

        If the key is not a dataclass field on GeoPriorConfig, the store
        will treat it as an "extra" GUI-only key (that's fine).
        """
        st = getattr(self.env, "store", None)
        if st is None or not hasattr(st, "set"):
            return
        try:
            st.set(key, value)
        except Exception:
            # Never let a store write break planning.
            pass

    # -----------------------------------------------------------------
    # Plan construction
    # -----------------------------------------------------------------
    def plan_from_gui(self, gui_state: TuneGuiState) -> Optional[TunePlan]:
        """
        Build and validate a TunePlan from the current GUI state.

        Resolution precedence (v3.2)
        ----------------------------
        - City:
            GUI text -> store("city") -> cfg.city -> cfg.CITY_NAME
        - Search space:
            GUI -> store("tuner_search_space") -> cfg.tuner_search_space
            -> default_tuner_search_space()
        - Max trials:
            GUI -> store("tuner_max_trials") -> cfg.tuner_max_trials
            -> cfg.TUNER_MAX_TRIALS -> 20
        - Overrides:
            build_cfg_overrides() (store snapshot first), then inject
            mandatory tuning keys.
        """
        # Store-first config resolution:
        # - if env.store exists, resolve_cfg() returns store.cfg
        # - else resolve_cfg() returns env.geo_cfg (from __post_init__)
        cfg = self.resolve_cfg()

        # -----------------------------
        # 1) Resolve the city name
        # -----------------------------
        city = (gui_state.city_text or "").strip()

        # If GUI is empty, try store and config fallbacks.
        if not city:
            city = str(self._store_get("city", "") or "").strip()
        if not city:
            city = str(getattr(cfg, "city", "") or "").strip()
        if not city:
            city = str(getattr(cfg, "CITY_NAME", "") or "").strip()

        # City is required for tuning because Stage-1 lookup is city-based.
        if not city:
            self.hooks.warn(
                "Missing city",
                "Please provide a city/dataset name, or pick one "
                "from the tuning options.",
            )
            return None

        # -----------------------------
        # 2) Resolve tuner search space
        # -----------------------------
        search_space = gui_state.tuner_search_space

        # Try store/cfg if GUI doesn't provide one.
        if not search_space:
            ss = self._store_get("tuner_search_space", None)
            if isinstance(ss, dict) and ss:
                search_space = ss

        if not search_space:
            ss = getattr(cfg, "tuner_search_space", None)
            if isinstance(ss, dict) and ss:
                search_space = ss

        # Final fallback: stable defaults shipped with the app.
        if not search_space:
            search_space = default_tuner_search_space()

        # -----------------------------
        # 3) Resolve max_trials
        # -----------------------------
        max_trials = gui_state.max_trials

        if max_trials is None:
            mt = self._store_get("tuner_max_trials", None)
            if mt is None:
                mt = getattr(cfg, "tuner_max_trials", None)
            if mt is None:
                mt = getattr(cfg, "TUNER_MAX_TRIALS", 20)

            # Convert to int defensively.
            try:
                max_trials = int(mt)
            except Exception:
                max_trials = 20
        else:
            max_trials = int(max_trials)

        # -----------------------------
        # 4) Mirror normalized values
        # -----------------------------
        # v3.2: store is primary (single source of truth)
        self._store_set("city", city)
        self._store_set("tuner_search_space", search_space)
        self._store_set("tuner_max_trials", max_trials)

        # v3.0: mirror into legacy config for backward compatibility.
        # These are best-effort; ignore failures.
        if hasattr(cfg, "city"):
            try:
                setattr(cfg, "city", city)
            except Exception:
                pass
        if hasattr(cfg, "tuner_search_space"):
            try:
                setattr(cfg, "tuner_search_space", search_space)
            except Exception:
                pass
        if hasattr(cfg, "TUNER_MAX_TRIALS"):
            try:
                setattr(cfg, "TUNER_MAX_TRIALS", max_trials)
            except Exception:
                pass

        # -----------------------------
        # 5) Build overrides map
        # -----------------------------
        # Store-first overrides snapshot + device overrides (GPU/CPU flags).
        overrides = self.build_cfg_overrides()

        # Ensure results are written into the GUI runs root.
        overrides.setdefault(
            "BASE_OUTPUT_DIR",
            str(Path(self.env.gui_runs_root)),
        )

        # Inject required tuning keys the backend expects.
        overrides["CITY_NAME"] = city
        overrides["TUNER_MAX_TRIALS"] = max_trials

        # Keep the search space available to the backend.
        # (If the backend expects a JSON string later, serialize there.)
        overrides.setdefault("TUNER_SEARCH_SPACE", search_space)

        return TunePlan(
            city=city,
            manifest_path=gui_state.manifest_path,
            stage1_root=gui_state.stage1_root,
            cfg_overrides=overrides,
            max_trials=max_trials,
        )

    # -----------------------------------------------------------------
    # Dry-run preview
    # -----------------------------------------------------------------
    def dry_preview(self, gui_state: TuneGuiState) -> None:
        """
        Preview the tuning plan without starting any threads.

        This is used when env.dry_mode=True to let users verify:
        - which city is targeted
        - which manifest is used / auto-discovered
        - max_trials and search space keys
        - the resolved results root
        """
        self.progress(0.0)
        self.status("Dry-run / Tune: resolving tuning job…")
        self.log("[Dry-run] Previewing GeoPrior tuning workflow...")

        plan = self.plan_from_gui(gui_state)
        if plan is None:
            # plan_from_gui already warned the user.
            self.progress(1.0)
            self.status(
                "Dry-run / Tune: preview aborted – invalid configuration."
            )
            return

        self.progress(0.35)
        self.status(
            f"Dry-run / Tune: syncing configuration for city={plan.city}…"
        )

        search_space = plan.cfg_overrides.get(
            "TUNER_SEARCH_SPACE",
            default_tuner_search_space(),
        )

        self.progress(0.65)

        results_root = Path(self.env.gui_runs_root)
        stage1_root = plan.stage1_root or "<from manifest / auto-discover>"
        manifest = plan.manifest_path or "<auto-discover>"

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

        self.progress(1.0)
        self.status("Dry-run / Tune: preview complete (100 %).")

    # -----------------------------------------------------------------
    # Real tuning run
    # -----------------------------------------------------------------
    def start_real_run(
        self,
        gui_state: TuneGuiState,
        start_tuning_cb: Callable[[TunePlan, bool], None],
    ) -> None:
        """
        Start a real tuning workflow.

        The GUI provides `start_tuning_cb`, which is responsible for:
        - instantiating and starting the TuningThread
        - wiring signals (progress, logs, completion)
        - managing UI state (disable buttons, timers, etc.)
        """
        if getattr(self.env, "dry_mode", False):
            self.dry_preview(gui_state)
            return

        plan = self.plan_from_gui(gui_state)
        if plan is None:
            return

        self.log(f"Start GeoPrior tuning for city={plan.city!r}.")
        self.status(
            f"Stage-2: tuning GeoPrior model for city={plan.city}."
        )
        self.progress(0.0)

        # Delegate thread creation and execution to the GUI layer.
        start_tuning_cb(plan, gui_state.eval_tuned)
