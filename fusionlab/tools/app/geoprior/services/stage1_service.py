# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>
#
# Stage-1 discovery / reuse logic extracted from the GeoPrior GUI.

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
)

from ..workflows.base import GUIHooks, RunEnv


@dataclass
class Stage1Decision:
    """
    Result of a smart Stage-1 handshake.

    need_stage1
        If True, Stage-1 should be (re)run now.
    manifest_hint
        Path to the Stage-1 manifest to reuse (if any).
    cancelled
        True if the user cancelled the choice dialog.
    messages
        Log lines describing how the decision was reached.
    """

    need_stage1: bool
    manifest_hint: Optional[str]
    cancelled: bool
    messages: List[str] = field(default_factory=list)


Stage1RunSummary = Any

Stage1Finder = Callable[
    [str, Path, Dict[str, Any]],
    Tuple[
        Sequence[Stage1RunSummary],
        Sequence[Stage1RunSummary],
    ],
]

Stage1Chooser = Callable[
    [str,
     Sequence[Stage1RunSummary],
     Sequence[Stage1RunSummary],
     bool],
    Tuple[str, Optional[Stage1RunSummary]],
]


class Stage1Service:
    """
    Encapsulate Stage-1 discovery and reuse logic.

    v3.2 store-first:
    - config reads use env.resolve_cfg()
    - flags prefer store.get(...) when available
    - GeoPriorConfig is fallback
    """

    def __init__(
        self,
        env: RunEnv,
        hooks: GUIHooks,
        find_stage1_for_city: Stage1Finder,
        chooser: Optional[Stage1Chooser] = None,
    ) -> None:
        self.env = env
        self.hooks = hooks
        self._finder = find_stage1_for_city
        self._chooser = chooser

    # -------------------------------------------------
    # Internals (store-first config reads)
    # -------------------------------------------------
    def _cfg_bool(
        self,
        name: str,
        default: bool,
    ) -> bool:
        st = getattr(self.env, "store", None)
        if st is not None and hasattr(st, "get"):
            try:
                return bool(st.get(name, default))
            except Exception:
                pass

        cfg = self.env.resolve_cfg()
        try:
            return bool(getattr(cfg, name, default))
        except Exception:
            return bool(default)

    def _log(
        self,
        messages: List[str],
        msg: str,
    ) -> None:
        messages.append(msg)
        try:
            self.hooks.log(msg)
        except Exception:
            pass

    @staticmethod
    def _manifest_path(obj: Any) -> Optional[str]:
        for a in (
            "manifest_path",
            "manifest",
            "path",
            "stage1_manifest",
        ):
            if hasattr(obj, a):
                try:
                    v = getattr(obj, a)
                    if v:
                        return str(v)
                except Exception:
                    continue
        return None

    @staticmethod
    def _ts_score(run: Any) -> float:
        v = getattr(run, "timestamp", None)
        if v is None:
            return 0.0

        if hasattr(v, "timestamp"):
            try:
                return float(v.timestamp())
            except Exception:
                pass

        try:
            return float(v)
        except Exception:
            pass

        try:
            s = str(v).strip()
            digits = "".join(ch for ch in s if ch.isdigit())
            return float(digits or "0")
        except Exception:
            return 0.0

    @staticmethod
    def _diff_msg(run: Any) -> str:
        diff_fields = getattr(run, "diff_fields", None)
        if not diff_fields:
            return ""
        try:
            txt = ", ".join(diff_fields)
            if txt.strip():
                return " (diff: " + txt + ")"
        except Exception:
            return ""
        return ""

    def _log_run_summary(
        self,
        messages: List[str],
        prefix: str,
        run: Any,
        *,
        city_fallback: str,
    ) -> None:
        city = getattr(run, "city", None) or city_fallback
        run_dir = getattr(run, "run_dir", "?")
        time_steps = getattr(run, "time_steps", "?")
        horizon = getattr(run, "horizon_years", "?")
        n_train = getattr(run, "n_train", "?")
        n_val = getattr(run, "n_val", "?")

        diff = self._diff_msg(run)

        self._log(
            messages,
            prefix + diff + ":\n"
            f"  City        : {city}\n"
            f"  Run dir     : {run_dir}\n"
            f"  T, H        : {time_steps}, {horizon}\n"
            f"  train / val : {n_train} / {n_val}",
        )

    def _latest_match(
        self,
        runs: Sequence[Any],
    ) -> Optional[Any]:
        cands = [
            r
            for r in runs
            if bool(getattr(r, "is_complete", False))
            and bool(getattr(r, "config_match", False))
        ]
        if not cands:
            return None
        return max(cands, key=self._ts_score)

    # -------------------------------------------------
    # Public API
    # -------------------------------------------------
    def decide(
        self,
        city: str,
        clean_stage1_dir: bool,
    ) -> Stage1Decision:
        messages: List[str] = []

        # 0) Forced rebuild via Training options
        if bool(clean_stage1_dir):
            self._log(
                messages,
                "[SmartStage1] 'Clean Stage-1 run dir' "
                "is enabled → forcing Stage-1 rebuild.",
            )
            return Stage1Decision(
                need_stage1=True,
                manifest_hint=None,
                cancelled=False,
                messages=messages,
            )

        # 1) Build current Stage-1 config snapshot
        cfg = self.env.resolve_cfg()
        try:
            current_cfg = cfg.to_stage1_config()
        except Exception as exc:
            self._log(
                messages,
                "[SmartStage1] Failed to build current "
                "Stage-1 config "
                f"({exc}) → falling back to full Stage-1.",
            )
            return Stage1Decision(
                need_stage1=True,
                manifest_hint=None,
                cancelled=False,
                messages=messages,
            )

        # 2) Discover Stage-1 runs under GUI results root
        results_root = Path(self.env.gui_runs_root)
        try:
            runs_city, all_runs = self._finder(
                city=city,
                results_root=results_root,
                current_cfg=current_cfg,
            )
        except Exception as exc:
            self._log(
                messages,
                "[SmartStage1] Error while discovering "
                "Stage-1 runs "
                f"({exc}) → falling back to full Stage-1.",
            )
            return Stage1Decision(
                need_stage1=True,
                manifest_hint=None,
                cancelled=False,
                messages=messages,
            )

        if not runs_city:
            self._log(
                messages,
                "[SmartStage1] No previous Stage-1 manifest "
                "found for this city → running Stage-1.",
            )
            return Stage1Decision(
                need_stage1=True,
                manifest_hint=None,
                cancelled=False,
                messages=messages,
            )

        if not all_runs:
            self._log(
                messages,
                "[Stage-1] No existing Stage-1 runs in this "
                "root – building Stage-1 from scratch.",
            )
            return Stage1Decision(
                need_stage1=True,
                manifest_hint=None,
                cancelled=False,
                messages=messages,
            )

        # Smart options (store-first)
        auto_reuse = self._cfg_bool(
            "stage1_auto_reuse_if_match",
            True,
        )
        force_mis = self._cfg_bool(
            "stage1_force_rebuild_if_mismatch",
            True,
        )

        # 3a) Auto-reuse: latest complete + config_match
        if auto_reuse:
            best = self._latest_match(runs_city)
            if best is not None:
                mp = self._manifest_path(best)
                ts = getattr(best, "timestamp", "?")

                if mp:
                    self._log(
                        messages,
                        "[Stage-1] Auto-reusing complete "
                        f"Stage-1 run for city '{city}' "
                        f"@ {ts} (config matches current "
                        "GUI settings).",
                    )
                    self._log_run_summary(
                        messages,
                        "[SmartStage1] Reusing Stage-1 run",
                        best,
                        city_fallback=city,
                    )
                    return Stage1Decision(
                        need_stage1=False,
                        manifest_hint=mp,
                        cancelled=False,
                        messages=messages,
                    )

                self._log(
                    messages,
                    "[Stage-1] Found a matching Stage-1 run "
                    "but manifest path is missing → "
                    "forcing Stage-1 rebuild.",
                )
                return Stage1Decision(
                    need_stage1=True,
                    manifest_hint=None,
                    cancelled=False,
                    messages=messages,
                )

        # 3b) Force rebuild when nothing matches config
        any_match = any(
            bool(getattr(r, "config_match", False))
            for r in runs_city
        )
        if force_mis and (not any_match):
            self._log(
                messages,
                "[Stage-1] Existing Stage-1 runs found but "
                "none match the current GUI config – forcing "
                f"Stage-1 rebuild for city '{city}'.",
            )
            return Stage1Decision(
                need_stage1=True,
                manifest_hint=None,
                cancelled=False,
                messages=messages,
            )

        # 3c) Interactive choice dialog
        if self._chooser is None:
            self._log(
                messages,
                "[SmartStage1] No Stage-1 choice hook "
                "provided – falling back to Stage-1 rebuild.",
            )
            return Stage1Decision(
                need_stage1=True,
                manifest_hint=None,
                cancelled=False,
                messages=messages,
            )

        decision, selected = self._chooser(
            city=city,
            runs_for_city=runs_city,
            all_runs=all_runs,
            clean_stage1=clean_stage1_dir,
        )
        decision = (decision or "").lower().strip()

        if decision == "cancel":
            self._log(
                messages,
                "[SmartStage1] Training cancelled by user.",
            )
            return Stage1Decision(
                need_stage1=False,
                manifest_hint=None,
                cancelled=True,
                messages=messages,
            )

        if decision == "rebuild":
            self._log(
                messages,
                "[SmartStage1] User requested Stage-1 rebuild.",
            )
            return Stage1Decision(
                need_stage1=True,
                manifest_hint=None,
                cancelled=False,
                messages=messages,
            )

        if decision == "reuse" and selected is not None:
            is_complete = bool(
                getattr(selected, "is_complete", False)
            )
            config_match = bool(
                getattr(selected, "config_match", False)
            )

            if (not is_complete) or (not config_match):
                msg = (
                    "The selected Stage-1 run is incomplete or "
                    "incompatible with the current configuration.\n\n"
                    "Stage-1 will be rebuilt."
                )
                try:
                    self.hooks.warn(
                        "Invalid Stage-1 selection",
                        msg,
                    )
                except Exception:
                    pass

                self._log(
                    messages,
                    "[SmartStage1] Selected Stage-1 run is "
                    "incomplete or incompatible → forcing rebuild.",
                )
                return Stage1Decision(
                    need_stage1=True,
                    manifest_hint=None,
                    cancelled=False,
                    messages=messages,
                )

            mp = self._manifest_path(selected)
            if not mp:
                self._log(
                    messages,
                    "[SmartStage1] Selected Stage-1 run is "
                    "missing manifest path → forcing rebuild.",
                )
                return Stage1Decision(
                    need_stage1=True,
                    manifest_hint=None,
                    cancelled=False,
                    messages=messages,
                )

            self._log_run_summary(
                messages,
                "[SmartStage1] Reusing Stage-1 run",
                selected,
                city_fallback=city,
            )

            return Stage1Decision(
                need_stage1=False,
                manifest_hint=mp,
                cancelled=False,
                messages=messages,
            )

        self._log(
            messages,
            "[SmartStage1] Unexpected decision="
            f"{decision!r} from Stage-1 choice dialog – "
            "falling back to Stage-1 rebuild.",
        )
        return Stage1Decision(
            need_stage1=True,
            manifest_hint=None,
            cancelled=False,
            messages=messages,
        )
