# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>
#
# Stage-1 discovery / reuse logic extracted from the GeoPrior GUI.
#
# This module provides a small, testable service around the
# "smart Stage-1 handshake" that used to live inside
# GeoPriorForecaster._smart_stage1_handshake.
#
# Key responsibilities
# --------------------
# - Inspect existing Stage-1 runs for a given city under a results root.
# - Decide whether to:
#       * run Stage-1 from scratch / rebuild, or
#       * reuse an existing Stage-1 manifest, or
#       * cancel (user choice).
# - Return a pure data object (Stage1Decision) that higher-level
#   controllers can use for:
#       * real training flows, or
#       * dry-run previews (no threads, no I/O).
#
# The service is intentionally UI-agnostic:
# all GUI interaction (dialogs, warnings) is delegated to `GUIHooks`
# supplied by the caller, and to a separate `Stage1Chooser` callback
# that wraps Stage1ChoiceDialog.ask at the GUI layer.

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

# reuse shared env / hooks from workflows.base
from ..workflows.base import RunEnv, GUIHooks

###############################################################################
# Public types
###############################################################################


@dataclass
class Stage1Decision:
    """
    Result of a smart Stage-1 handshake.

    Parameters
    ----------
    need_stage1 : bool
        If True, Stage-1 should be (re)run now.
        If False and `cancelled` is False, an existing Stage-1 run
        can be reused via `manifest_hint`.
    manifest_hint : str or None
        Path to the Stage-1 manifest to reuse, if any. Only meaningful
        when ``need_stage1`` is False and ``cancelled`` is False.
    cancelled : bool
        True if the user cancelled in the Stage-1 choice dialog.
        In this case, callers should abort the workflow.
    messages : list of str
        Log lines describing how the decision was reached. The caller
        is free to forward them to a GUI log widget or logger.
    """

    need_stage1: bool
    manifest_hint: Optional[str]
    cancelled: bool
    messages: List[str] = field(default_factory=list)


# A Stage-1 run summary is an opaque object from the backend. We only
# rely on a few attributes (documented below).
Stage1RunSummary = Any

# Signature for a discovery helper, typically backed by the existing
# NATCOM / GeoPrior utility that scans the results tree.
Stage1Finder = Callable[
    [str, Path, Dict[str, Any]],
    Tuple[Sequence[Stage1RunSummary], Sequence[Stage1RunSummary]],
]

# Signature for the Stage-1 choice dialog (Stage1ChoiceDialog.ask)
Stage1Chooser = Callable[
    [str, Sequence[Stage1RunSummary], Sequence[Stage1RunSummary], bool],
    Tuple[str, Optional[Stage1RunSummary]],
]


###############################################################################
# Service implementation
###############################################################################


class Stage1Service:
    """
    Encapsulate Stage-1 discovery and reuse logic.

    This service mirrors the behaviour of the original
    ``_smart_stage1_handshake`` method from GeoPriorForecaster, but:

    - keeps all logic in a pure, testable class;
    - does not depend on Qt or widgets;
    - returns a :class:`Stage1Decision` data object.

    Parameters
    ----------
    env : RunEnv
        Runtime environment carrying the GeoPriorConfig and the
        GUI runs root. This is the shared RunEnv from
        :mod:`geoprior.workflows.base` (geo_cfg, gui_runs_root, etc.).
    hooks : GUIHooks
        Shared GUI callbacks (log, warn, error, etc.) from
        :mod:`geoprior.workflows.base`. Stage1Service currently uses:
            - hooks.log(title)
            - hooks.warn(title, msg)
    find_stage1_for_city : callable
        Backend helper used to list Stage-1 runs.

        It must accept the arguments::

            runs_for_city, all_runs = find_stage1_for_city(
                city=city,
                results_root=results_root,
                current_cfg=current_cfg,
            )

        where ``runs_for_city`` and ``all_runs`` are sequences of
        Stage1RunSummary objects exposing at least:

            - .city
            - .timestamp
            - .run_dir
            - .is_complete  (bool)
            - .config_match (bool)
            - .manifest_path
            - .time_steps
            - .horizon_years
            - .n_train
            - .n_val
            - .diff_fields (iterable of str, optional)
    chooser : callable or None, optional
        Function implementing the Stage-1 choice dialog, typically
        wrapping ``Stage1ChoiceDialog.ask`` from the GUI layer.

        Expected signature::

            decision, selected = chooser(
                city=city,
                runs_for_city=runs_for_city,
                all_runs=all_runs,
                clean_stage1=clean_stage1_dir,
            )

        where:

        - ``decision`` is one of {"cancel", "rebuild", "reuse"}.
        - ``selected`` is a Stage1RunSummary or ``None``.

        If ``chooser`` is None, the service will never ask the user and
        will fall back to the safest behaviour (rebuild Stage-1) when
        interaction would be required.
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

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def decide(
        self,
        city: str,
        clean_stage1_dir: bool,
    ) -> Stage1Decision:
        """
        Decide whether Stage-1 should be (re)run or reused.

        This implements the same semantics as the legacy
        ``_smart_stage1_handshake``:

        1. If ``clean_stage1_dir`` is True → force Stage-1 rebuild.
        2. Otherwise, build a Stage-1 config snapshot and look for
           matching runs under ``env.gui_runs_root``.
        3. If none exist for this city → run Stage-1.
        4. If global directory is empty → run Stage-1.
        5. If ``stage1_auto_reuse_if_match`` is True, auto-reuse the
           latest complete + config_match run.
        6. If ``stage1_force_rebuild_if_mismatch`` is True and no
           run matches the current config, force rebuild.
        7. Otherwise, ask the user via the Stage-1 choice dialog
           (``Stage1Chooser``).

        Returns
        -------
        Stage1Decision
            Data object describing whether Stage-1 must be run,
            which manifest to reuse (if any), and whether the
            user cancelled the workflow.
        """
        messages: List[str] = []

        def _log(msg: str) -> None:
            messages.append(msg)
            try:
                self.hooks.log(msg)
            except Exception:
                # Logging should never break the decision logic
                pass

        geo_cfg = self.env.geo_cfg

        # --------------------------------------------------------------
        # 0) Forced rebuild via Training options
        # --------------------------------------------------------------
        if bool(clean_stage1_dir):
            _log(
                "[SmartStage1] 'Clean Stage-1 run dir' is enabled → "
                "forcing Stage-1 rebuild."
            )
            return Stage1Decision(
                need_stage1=True,
                manifest_hint=None,
                cancelled=False,
                messages=messages,
            )

        # --------------------------------------------------------------
        # 1) Build current Stage-1 config snapshot
        # --------------------------------------------------------------
        try:
            current_cfg: Dict[str, Any] = geo_cfg.to_stage1_config()
        except Exception as exc:  # pragma: no cover - defensive
            _log(
                "[SmartStage1] Failed to build current Stage-1 config "
                f"({exc}) → falling back to full Stage-1."
            )
            return Stage1Decision(
                need_stage1=True,
                manifest_hint=None,
                cancelled=False,
                messages=messages,
            )

        # --------------------------------------------------------------
        # 2) Discover Stage-1 manifests under the GUI results root
        # --------------------------------------------------------------
        results_root = Path(self.env.gui_runs_root)

        try:
            runs_for_city, all_runs = self._finder(
                city=city,
                results_root=results_root,
                current_cfg=current_cfg,
            )
        except Exception as exc:  # pragma: no cover - defensive
            _log(
                "[SmartStage1] Error while discovering Stage-1 runs "
                f"({exc}) → falling back to full Stage-1."
            )
            return Stage1Decision(
                need_stage1=True,
                manifest_hint=None,
                cancelled=False,
                messages=messages,
            )

        if not runs_for_city:
            _log(
                "[SmartStage1] No previous Stage-1 manifest found for "
                "this city → running Stage-1."
            )
            return Stage1Decision(
                need_stage1=True,
                manifest_hint=None,
                cancelled=False,
                messages=messages,
            )

        if not all_runs:
            # Should be rare, but kept for backward compatibility
            _log(
                "[Stage-1] No existing Stage-1 runs in this root – "
                "building Stage-1 from scratch."
            )
            return Stage1Decision(
                need_stage1=True,
                manifest_hint=None,
                cancelled=False,
                messages=messages,
            )

        # Smart options from config
        auto_reuse = bool(
            getattr(geo_cfg, "stage1_auto_reuse_if_match", True)
        )
        force_rebuild_mismatch = bool(
            getattr(geo_cfg, "stage1_force_rebuild_if_mismatch", True)
        )

        # --------------------------------------------------------------
        # 3a) Auto-reuse path: latest complete + config_match
        # --------------------------------------------------------------
        if auto_reuse:
            best = next(
                (
                    r
                    for r in runs_for_city
                    if getattr(r, "is_complete", False)
                    and getattr(r, "config_match", False)
                ),
                None,
            )
            if best is not None:
                ts = getattr(best, "timestamp", "?")
                _log(
                    "[Stage-1] Auto-reusing complete Stage-1 run "
                    f"for city '{city}' @ {ts} "
                    "(config matches current GUI settings)."
                )
                manifest_hint = str(getattr(best, "manifest_path", ""))
                return Stage1Decision(
                    need_stage1=False,
                    manifest_hint=manifest_hint or None,
                    cancelled=False,
                    messages=messages,
                )

        # --------------------------------------------------------------
        # 3b) Force rebuild when nothing matches the current config
        # --------------------------------------------------------------
        any_match = any(
            getattr(r, "config_match", False) for r in runs_for_city
        )
        if force_rebuild_mismatch and not any_match:
            _log(
                "[Stage-1] Existing Stage-1 runs found but none match "
                f"the current GUI config – forcing Stage-1 rebuild for "
                f"city '{city}'."
            )
            return Stage1Decision(
                need_stage1=True,
                manifest_hint=None,
                cancelled=False,
                messages=messages,
            )

        # --------------------------------------------------------------
        # 3c) Fallback: interactive Stage-1 choice dialog
        # --------------------------------------------------------------
        if self._chooser is None:
            # No dialog hook: safest fallback is to rebuild
            _log(
                "[SmartStage1] No Stage-1 choice hook provided – "
                "falling back to Stage-1 rebuild."
            )
            return Stage1Decision(
                need_stage1=True,
                manifest_hint=None,
                cancelled=False,
                messages=messages,
            )

        decision, selected = self._chooser(
            city=city,
            runs_for_city=runs_for_city,
            all_runs=all_runs,
            clean_stage1=clean_stage1_dir,
        )

        decision = (decision or "").lower().strip()

        if decision == "cancel":
            _log("[SmartStage1] Training cancelled by user.")
            return Stage1Decision(
                need_stage1=False,
                manifest_hint=None,
                cancelled=True,
                messages=messages,
            )

        if decision == "rebuild":
            _log("[SmartStage1] User requested Stage-1 rebuild.")
            return Stage1Decision(
                need_stage1=True,
                manifest_hint=None,
                cancelled=False,
                messages=messages,
            )

        if decision == "reuse" and selected is not None:
            # Safety: only reuse fully compatible + complete runs
            is_complete = bool(
                getattr(selected, "is_complete", False)
            )
            config_match = bool(
                getattr(selected, "config_match", False)
            )

            if (not is_complete) or (not config_match):
                # Mirror the old QMessageBox warning via hooks.warn
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
                    # Warning dialog failures should not crash the flow
                    pass

                _log(
                    "[SmartStage1] Selected Stage-1 run is incomplete "
                    "or incompatible → forcing rebuild."
                )
                return Stage1Decision(
                    need_stage1=True,
                    manifest_hint=None,
                    cancelled=False,
                    messages=messages,
                )

            # At this point we can safely reuse the run
            manifest_hint = str(
                getattr(selected, "manifest_path", "")
            ) or None

            diff_fields = getattr(selected, "diff_fields", None)
            diff_msg = ""
            if diff_fields:
                try:
                    diff_msg = " (diff: " + ", ".join(diff_fields) + ")"
                except Exception:
                    # diff_fields might not be iterable; ignore gracefully
                    diff_msg = ""

            city_sel = getattr(selected, "city", city)
            run_dir = getattr(selected, "run_dir", "?")
            time_steps = getattr(selected, "time_steps", "?")
            horizon = getattr(selected, "horizon_years", "?")
            n_train = getattr(selected, "n_train", "?")
            n_val = getattr(selected, "n_val", "?")

            _log(
                "[SmartStage1] Reusing Stage-1 run"
                + diff_msg
                + ":\n"
                f"  City        : {city_sel}\n"
                f"  Run dir     : {run_dir}\n"
                f"  T, H        : {time_steps}, {horizon}\n"
                f"  train / val : {n_train} / {n_val}"
            )

            return Stage1Decision(
                need_stage1=False,
                manifest_hint=manifest_hint,
                cancelled=False,
                messages=messages,
            )

        # Fallback: if the dialog returned something unexpected, be safe
        _log(
            f"[SmartStage1] Unexpected decision={decision!r} from "
            "Stage-1 choice dialog – falling back to Stage-1 rebuild."
        )
        return Stage1Decision(
            need_stage1=True,
            manifest_hint=None,
            cancelled=False,
            messages=messages,
        )
