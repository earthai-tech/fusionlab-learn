# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>
#
# geoprior/workflows/transfer.py
#
# Transferability workflow controller (Xfer matrix).
#
# v3.2 goals
# ----------
# - Store-first: if env.store exists, read/write there first.
# - Back-compat: if no store, fall back to env.resolve_cfg().
# - Qt-free: no widget imports; UI goes through GUIHooks and
#   a GUI callback that starts XferMatrixThread.
#
# Note on GeoConfigStore "extra" keys
# -----------------------------------
# GeoConfigStore.set("xfer.*", ...) is safe even if "xfer.*"
# is not a GeoPriorConfig dataclass field: the store will keep
# it in its GUI-only extra dict and still emit change signals.

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, List, Optional, Sequence

from .base import BaseWorkflowController, GUIHooks, RunEnv


# ---------------------------------------------------------------------
# Data structures (Qt-free)
# ---------------------------------------------------------------------


@dataclass
class TransferGuiState:
    """
    Snapshot of the Transferability tab (Qt-free).

    The GUI should provide plain Python types only.
    Any field can be left empty if the controller can
    fill it from store/cfg defaults (dry preview).
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

    # ---- v3.2 extras (optional) ----
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
    Validated plan passed to XferMatrixThread.

    Keep this stable: the GUI callback and threads can
    rely on these fields without importing Qt.
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


StartXferCb = Callable[[TransferPlan], None]


# ---------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------


class TransferController(BaseWorkflowController):
    """
    Plan + dry-run logic for cross-city transfer matrix.

    Responsibilities
    ----------------
    - Resolve missing values from store/cfg (store first).
    - Validate required inputs (cities, splits, calib modes).
    - Provide a dry preview (logs + progress only).
    - Hand a TransferPlan to a GUI callback for real runs.
    """

    def __init__(self, env: RunEnv, hooks: GUIHooks) -> None:
        super().__init__(env, hooks)

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
    # Small normalization helpers
    # -----------------------------------------------------------------
    @staticmethod
    def _clean_list(items: Optional[Sequence[str]]) -> List[str]:
        out: List[str] = []
        for x in (items or []):
            s = str(x or "").strip()
            if s:
                out.append(s)
        return out

    @staticmethod
    def _clean_text(x: Any) -> str:
        return str(x or "").strip()

    @staticmethod
    def _as_path(x: Any) -> Optional[Path]:
        s = str(x or "").strip()
        if not s:
            return None
        return Path(s)

    @staticmethod
    def _as_float_list(
        x: Optional[Sequence[float]],
    ) -> Optional[List[float]]:
        if x is None:
            return None
        out: List[float] = []
        for v in x:
            try:
                out.append(float(v))
            except Exception:
                continue
        return out if out else None

    # -----------------------------------------------------------------
    # Plan builder
    # -----------------------------------------------------------------
    def _build_plan(
        self,
        state: TransferGuiState,
        *,
        for_dry_run: bool,
    ) -> Optional[TransferPlan]:
        """
        Turn GUI snapshot into a validated TransferPlan.

        Precedence (v3.2)
        -----------------
        - cities:
            GUI -> store("xfer.city_a/b") -> cfg fallback
            (dry-run only) -> default nansha/zhongshan
        - results_root:
            GUI -> store("xfer.results_root") -> env.gui_runs_root
        - splits / calib_modes:
            GUI -> store("xfer.splits"/"xfer.calib_modes")
        - other flags:
            GUI values are taken as authoritative, but we still
            mirror them into the store for persistence.
        """
        cfg = self.resolve_cfg()

        # -----------------------------
        # results root
        # -----------------------------
        root = self._as_path(state.results_root)
        if root is None:
            root = self._as_path(
                self._store_get("xfer.results_root", None)
            )
        if root is None:
            root = Path(self.env.gui_runs_root)

        # -----------------------------
        # cities (A/B)
        # -----------------------------
        city_a = self._clean_text(state.city_a)
        city_b = self._clean_text(state.city_b)

        if not city_a:
            city_a = self._clean_text(
                self._store_get("xfer.city_a", "")
            )
        if not city_b:
            city_b = self._clean_text(
                self._store_get("xfer.city_b", "")
            )

        # Optional cfg fallbacks (rare, but safe)
        if not city_a:
            city_a = self._clean_text(
                getattr(cfg, "xfer_city_a", "")
            )
        if not city_b:
            city_b = self._clean_text(
                getattr(cfg, "xfer_city_b", "")
            )

        # Dry preview convenience defaults (old behaviour)
        if for_dry_run:
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

        if city_a == city_b:
            self.hooks.warn(
                "Invalid cities",
                "City A and City B must be different.",
            )
            self.progress(0.0)
            return None

        # -----------------------------
        # splits / calib modes
        # -----------------------------
        splits = self._clean_list(state.splits)
        if not splits:
            splits = self._clean_list(
                self._store_get("xfer.splits", [])
            )
        if not splits:
            self.hooks.warn(
                "Splits required",
                "Please select at least one split.",
            )
            self.progress(0.0)
            return None

        calib = self._clean_list(state.calib_modes)
        if not calib:
            calib = self._clean_list(
                self._store_get("xfer.calib_modes", [])
            )
        if not calib:
            self.hooks.warn(
                "Calibration modes required",
                "Please select at least one mode.",
            )
            self.progress(0.0)
            return None

        # -----------------------------
        # optional lists
        # -----------------------------
        rescale_modes = state.rescale_modes
        if rescale_modes is None:
            rescale_modes = self._store_get(
                "xfer.rescale_modes",
                None,
            )
        rescale_modes = (
            self._clean_list(rescale_modes)
            if rescale_modes is not None
            else None
        )

        strategies = state.strategies
        if strategies is None:
            strategies = self._store_get(
                "xfer.strategies",
                None,
            )
        strategies = (
            self._clean_list(strategies)
            if strategies is not None
            else None
        )

        # -----------------------------
        # quantiles override
        # -----------------------------
        q_override = self._as_float_list(
            state.quantiles_override
        )

        # -----------------------------
        # align policy
        # -----------------------------
        align_policy = self._clean_text(state.align_policy)
        if not align_policy:
            align_policy = "align_by_name_pad"

        # -----------------------------
        # build plan
        # -----------------------------
        plan = TransferPlan(
            city_a=city_a,
            city_b=city_b,
            results_root=root,
            splits=splits,
            calib_modes=calib,
            rescale_to_source=bool(state.rescale_to_source),
            batch_size=int(state.batch_size),
            quantiles_override=q_override,
            write_json=bool(state.write_json),
            write_csv=bool(state.write_csv),
            rescale_modes=rescale_modes or None,
            strategies=strategies or None,
            prefer_tuned=bool(state.prefer_tuned),
            align_policy=align_policy,
            allow_reorder_dynamic=state.allow_reorder_dynamic,
            allow_reorder_future=state.allow_reorder_future,
            warm_split=state.warm_split,
            warm_samples=state.warm_samples,
            warm_frac=state.warm_frac,
            warm_epochs=state.warm_epochs,
            warm_lr=state.warm_lr,
            warm_seed=state.warm_seed,
        )

        # -----------------------------
        # mirror into store (v3.2)
        # -----------------------------
        self._store_set("xfer.city_a", plan.city_a)
        self._store_set("xfer.city_b", plan.city_b)
        self._store_set("xfer.results_root", str(plan.results_root))
        self._store_set("xfer.splits", list(plan.splits))
        self._store_set("xfer.calib_modes", list(plan.calib_modes))

        self._store_set(
            "xfer.rescale_to_source",
            bool(plan.rescale_to_source),
        )
        self._store_set("xfer.batch_size", int(plan.batch_size))
        self._store_set(
            "xfer.quantiles_override",
            plan.quantiles_override,
        )
        self._store_set("xfer.write_json", bool(plan.write_json))
        self._store_set("xfer.write_csv", bool(plan.write_csv))

        self._store_set("xfer.rescale_modes", plan.rescale_modes)
        self._store_set("xfer.strategies", plan.strategies)

        self._store_set("xfer.prefer_tuned", bool(plan.prefer_tuned))
        self._store_set("xfer.align_policy", plan.align_policy)

        self._store_set(
            "xfer.allow_reorder_dynamic",
            plan.allow_reorder_dynamic,
        )
        self._store_set(
            "xfer.allow_reorder_future",
            plan.allow_reorder_future,
        )

        self._store_set("xfer.warm_split", plan.warm_split)
        self._store_set("xfer.warm_samples", plan.warm_samples)
        self._store_set("xfer.warm_frac", plan.warm_frac)
        self._store_set("xfer.warm_epochs", plan.warm_epochs)
        self._store_set("xfer.warm_lr", plan.warm_lr)
        self._store_set("xfer.warm_seed", plan.warm_seed)

        return plan

    # -----------------------------------------------------------------
    # Dry preview
    # -----------------------------------------------------------------
    def dry_preview(self, state: TransferGuiState) -> None:
        """
        Simulate a transfer run without starting threads.

        Keeps the legacy convenience:
        - if cities are blank, default to nansha/zhongshan
          (dry-run only).
        """
        self.progress(0.0)
        self.status(
            "[DRY] Transfer preview – nothing will execute."
        )

        plan = self._build_plan(state, for_dry_run=True)
        if plan is None:
            return

        self.progress(0.5)

        q_display = (
            plan.quantiles_override
            if plan.quantiles_override is not None
            else "<from model>"
        )

        lines = [
            "[DRY] Transfer matrix plan:",
            f"  city_a        : {plan.city_a}",
            f"  city_b        : {plan.city_b}",
            f"  results_root  : {plan.results_root}",
            f"  splits        : {plan.splits}",
            f"  calib_modes   : {plan.calib_modes}",
            f"  rescale       : {plan.rescale_to_source}",
            f"  batch_size    : {plan.batch_size}",
            f"  quantiles     : {q_display}",
            f"  write_json    : {plan.write_json}",
            f"  write_csv     : {plan.write_csv}",
            f"  strategies    : {plan.strategies or '<default>'}",
            f"  rescale_modes : {plan.rescale_modes or '<auto>'}",
            f"  prefer_tuned  : {plan.prefer_tuned}",
            f"  align_policy  : {plan.align_policy}",
            f"  allow_re_dyn  : {plan.allow_reorder_dynamic}",
            f"  allow_re_fut  : {plan.allow_reorder_future}",
            f"  warm_split    : {plan.warm_split}",
            f"  warm_samples  : {plan.warm_samples}",
            f"  warm_frac     : {plan.warm_frac}",
            f"  warm_epochs   : {plan.warm_epochs}",
            f"  warm_lr       : {plan.warm_lr}",
            f"  warm_seed     : {plan.warm_seed}",
        ]
        for line in lines:
            self.log(line)

        self.progress(1.0)
        self.status(
            "[DRY] Transfer preview complete – no run started."
        )

    # -----------------------------------------------------------------
    # Real run
    # -----------------------------------------------------------------
    def start_real_run(
        self,
        state: TransferGuiState,
        start_xfer_cb: StartXferCb,
    ) -> None:
        """
        Validate GUI state and start a real transfer run.

        The GUI callback must:
        - construct XferMatrixThread(plan)
        - wire signals
        - start the thread
        """
        if getattr(self.env, "dry_mode", False):
            self.dry_preview(state)
            return

        plan = self._build_plan(state, for_dry_run=False)
        if plan is None:
            return

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

        start_xfer_cb(plan)
