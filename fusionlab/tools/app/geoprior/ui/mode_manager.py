# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>
#
# Centralised handler for:
#   - the "Mode: ..." badge (Train / Tune / Infer / Xfer / Results / Dry),
#   - the global Stop button (visibility + pulse),
#   - tooltips + icons of all "Run" buttons in normal vs Dry mode.
#
# The GeoPriorForecaster only forwards state changes to this helper.
#

from __future__ import annotations

from typing import Optional, Dict, Tuple

from PyQt5.QtCore import QObject, pyqtSignal, QTimer
from PyQt5.QtWidgets import QPushButton, QTabWidget

from ..styles import (
    MODE_DRY_COLOR,
    MODE_TRAIN_COLOR,
    MODE_TUNE_COLOR,
    MODE_INFER_COLOR,
    MODE_XFER_COLOR,
    MODE_RESULTS_COLOR,
)


HelpMap = Dict[str, Tuple[int, str]]
RunButtonMap = Dict[str, QPushButton]


class ModeManager(QObject):
    """
    Encapsulate all 'Mode' badge + Stop button behaviour.

    The main window should:
    - construct this once,
    - register its run buttons via :meth:`set_run_buttons`,
    - call :meth:`set_dry_mode` when the Dry checkbox toggles,
    - call :meth:`set_active_job_kind` when a job starts/ends,
    - call :meth:`update_for_tab` on tab change,
    - call :meth:`update_running_state` whenever jobs start/finish.

    Actual *cancelling* of threads is left to the main window:
    we only emit :pyattr:`stop_requested` when the Stop button is
    clicked while something is running.
    """

    stop_requested = pyqtSignal()


    # Centralised default help strings (used if caller does not override).
    DEFAULT_HELP_TEXTS: Dict[str, str] = {
        "train": (
            "Train tab – runs Stage-1 (prepare) and Stage-2 "
            "(GeoPrior training) for the selected city."
        ),
        "tune": (
            "Tune tab – runs Stage-2 hyperparameter search for the "
            "selected city using existing Stage-1 sequences.\n"
            "Note: Stage-1 must already exist for this city; run the "
            "Train tab first if necessary."
        ),
        "infer": (
            "Inference tab – evaluate a trained/tuned GeoPriorSubsNet on\n "
            "train/val/test splits or run future forecasts based on\n "
            "Stage-1 future NPZ artifacts."
        ),
        "xfer": (
            "Transferability tab – run cross-city transfer matrix (A↔B)\n "
            "and build view figures from xfer_results.* files."
        ),
        "results": (
            "Results tab – browse and download Stage-1 artifacts,\n"
            "train/tune/inference runs, and transferability outputs\n"
            "as ZIP archives."
        ),
    }

    def __init__(
        self,
        mode_btn: QPushButton,
        btn_stop: QPushButton,
        make_play_icon,
        stop_pulse_timer: QTimer,
        parent: Optional[QObject] = None,
    ) -> None:
        """
        Parameters
        ----------
        mode_btn : QPushButton
            Top "Mode: ..." badge.
        btn_stop : QPushButton
            Global Stop button.
        make_play_icon : callable
            Factory ``f(hollow: bool) -> QIcon`` used to build
            normal vs dry-run play icons.
        stop_pulse_timer : QTimer
            Timer used to drive a simple pulse animation on Stop.
        parent : QObject, optional
            Qt parent.
        """
        super().__init__(parent)

        self._mode_btn = mode_btn
        self._btn_stop = btn_stop
        self._make_play_icon = make_play_icon
        self._stop_pulse_timer = stop_pulse_timer

        # Run buttons are registered later
        self._run_buttons: RunButtonMap = {}
        self._tabs: Optional[QTabWidget] = None
        self._help_texts: HelpMap = {}

        # Internal state
        self._is_dry_mode: bool = False
        self._active_job_kind: Optional[str] = None  # "train", "tune", "infer", "xfer"
        self._any_running: bool = False
        self._stop_pulse_state: bool = False

        # Base + pulse styles for the Stop button
        self._stop_base_style: str = btn_stop.styleSheet() or ""
        self._stop_pulse_style: str = self._build_pulse_style(
            self._stop_base_style
        )

        if self._stop_pulse_timer is not None:
            self._stop_pulse_timer.timeout.connect(self._on_stop_pulse_tick)

    # ------------------------------------------------------------------
    # Registration helpers
    # ------------------------------------------------------------------
    def set_run_buttons(
        self,
        train_btn: Optional[QPushButton] = None,
        tune_btn: Optional[QPushButton] = None,
        infer_btn: Optional[QPushButton] = None,
        xfer_btn: Optional[QPushButton] = None,
    ) -> None:
        """
        Register the main 'Run' buttons so their icons/tooltips can be
        updated based on mode & running state.
        """
        mapping: RunButtonMap = {}
        if train_btn is not None:
            mapping["train"] = train_btn
        if tune_btn is not None:
            mapping["tune"] = tune_btn
        if infer_btn is not None:
            mapping["infer"] = infer_btn
        if xfer_btn is not None:
            mapping["xfer"] = xfer_btn

        self._run_buttons = mapping

    # ------------------------------------------------------------------
    # Public state setters (called by main window)
    # ------------------------------------------------------------------
    def set_dry_mode(self, is_dry: bool) -> None:
        """
        Update the internal Dry-run flag.

        The caller is responsible for invoking
        :meth:`update_for_tab` and :meth:`update_running_state`
        afterwards so the UI reflects the change.
        """
        self._is_dry_mode = bool(is_dry)

    def set_active_job_kind(self, kind: Optional[str]) -> None:
        """
        Record which job is currently active.

        Expected values
        ---------------
        None, "train", "tune", "infer", "xfer".
        """
        self._active_job_kind = kind

    # ------------------------------------------------------------------
    # Main update entrypoints
    # ------------------------------------------------------------------
    def _resolve_help_entry(
        self,
        key: str,
        default_idx: int,
        default_help: str,
    ) -> Tuple[int, str]:
        """
        Robustly extract (tab_index, help_text) from self._help_texts.
    
        Accepts:
        - (idx, help)
        - (idx, help, ...)   -> only first two are used
        - (idx,)             -> idx + default_help
        - "some help string" -> default_idx + that string
        - missing / None     -> (default_idx, default_help)
        """
        val = self._help_texts.get(key) if self._help_texts else None
        if val is None:
            return default_idx, default_help
    
        if isinstance(val, (tuple, list)):
            if len(val) >= 2:
                return int(val[0]), str(val[1])
            if len(val) == 1:
                return int(val[0]), default_help
            return default_idx, default_help
    
        # plain string or other scalar: treat as help text
        return default_idx, str(val)

    def update_for_tab(
        self,
        index: int,
        tabs: QTabWidget,
        help_texts: HelpMap,
    ) -> None:
        """
        Update the Mode badge for a newly selected tab.

        Parameters
        ----------
        index : int
            Current tab index.
        tabs : QTabWidget
            The tab widget (used to get tab text for fallback).
        help_texts : dict
            Mapping::

                {
                    "train":   (train_index,   train_help_text),
                    "tune":    (tune_index,    tune_help_text),
                    "infer":   (infer_index,   infer_help_text),
                    "xfer":    (xfer_index,    xfer_help_text),
                    "results": (results_index, results_help_text),
                }

            Any missing entries fall back to defaults.
        """
        self._tabs = tabs
        self._help_texts = help_texts or {}

        if self._mode_btn is None or tabs is None:
            return

        # If Dry-run is ON, it overrides per-tab modes
        if self._is_dry_mode:
            mode_label = "DRY RUN"
            color = MODE_DRY_COLOR
            tooltip = (
                "Dry-run mode – prepare configuration and log actions\n"
                "without actually running Stage-1 / Stage-2 / Stage-3."
            )
        else:
            # Extract mapping from help_texts
            # Extract mapping from help_texts (falling back to defaults)
            train_idx, train_help = self._resolve_help_entry(
                "train",
                -1,
                self.DEFAULT_HELP_TEXTS["train"],
            )
            tune_idx, tune_help = self._resolve_help_entry(
                "tune",
                -1,
                self.DEFAULT_HELP_TEXTS["tune"],
            )
            infer_idx, infer_help = self._resolve_help_entry(
                "infer",
                -1,
                self.DEFAULT_HELP_TEXTS["infer"],
            )
            xfer_idx, xfer_help = self._resolve_help_entry(
                "xfer",
                -1,
                self.DEFAULT_HELP_TEXTS["xfer"],
            )
            results_idx, results_help = self._resolve_help_entry(
                "results",
                -1,
                self.DEFAULT_HELP_TEXTS["results"],
            )

            if index == train_idx:
                mode_label = "TRAIN"
                color = MODE_TRAIN_COLOR
                tooltip = train_help
            elif index == tune_idx:
                mode_label = "TUNING"
                color = MODE_TUNE_COLOR
                tooltip = tune_help
            elif index == infer_idx:
                mode_label = "INFER"
                color = MODE_INFER_COLOR
                tooltip = infer_help
            elif index == xfer_idx:
                mode_label = "TRANSFER"
                color = MODE_XFER_COLOR
                tooltip = xfer_help
            elif index == results_idx:
                mode_label = "RESULTS"
                color = MODE_RESULTS_COLOR
                tooltip = results_help
            else:
                # Fallback: just use tab text
                tab_text = tabs.tabText(index) or "–"
                mode_label = tab_text
                color = MODE_TRAIN_COLOR
                tooltip = f"Mode: {tab_text}"

        # Apply label + tooltip + background color
        self._mode_btn.setText(f"Mode: {mode_label}")
        self._mode_btn.setToolTip(tooltip)
        self._mode_btn.setStyleSheet(
            f"""
            QPushButton {{
                background-color: {color};
                color: white;
                border-radius: 10px;
                padding: 2px 12px;
                font-weight: 600;
            }}
            QPushButton:disabled {{
                background-color: {color};
                color: white;
            }}
            """
        )

    def update_running_state(self, any_running: bool) -> None:
        """
        Refresh Stop button + run-button tooltips/icons based on:

        - at least one job running (any_running),
        - which job is active (self._active_job_kind),
        - whether Dry-run is enabled (self._is_dry_mode).
        """
        self._any_running = bool(any_running)

        # --- Stop button visibility + pulse ------------------------------
        if self._btn_stop is not None:
            self._btn_stop.setVisible(self._any_running)
            self._btn_stop.setEnabled(self._any_running)

            if self._any_running and self._stop_pulse_timer is not None:
                if not self._stop_pulse_timer.isActive():
                    self._stop_pulse_state = False
                    self._stop_pulse_timer.start()
            else:
                if self._stop_pulse_timer is not None:
                    self._stop_pulse_timer.stop()
                # Reset style when idle
                self._btn_stop.setStyleSheet(self._stop_base_style)

        # --- Run button tooltips + icons --------------------------------
        self._update_run_button_tooltips()
        self._update_run_button_icons()


    # ------------------------------------------------------------------
    # Slots for buttons / timer
    # ------------------------------------------------------------------
    def on_stop_clicked(self) -> None:
        """
        Invoked when the user clicks the global Stop button.

        We only handle UI aspects here and emit :pyattr:`stop_requested`
        so the main window can actually cancel running threads.
        """
        if not self._any_running:
            return

        # Prevent repeated clicks; will be re-enabled on next run
        if self._btn_stop is not None:
            self._btn_stop.setEnabled(False)

        # Let the owner take care of interrupting threads
        self.stop_requested.emit()

    # This is connected internally to _stop_pulse_timer.timeout
    def _on_stop_pulse_tick(self) -> None:
        """
        Simple pulse animation for the Stop button while jobs
        are running. The timer itself is started/stopped in
        :meth:`update_running_state`.
        """
        if self._btn_stop is None:
            return
        if not self._any_running:
            # Safety; the timer will normally be stopped by
            # update_running_state when any_running=False.
            if self._stop_pulse_timer is not None:
                self._stop_pulse_timer.stop()
            self._btn_stop.setStyleSheet(self._stop_base_style)
            return

        # Toggle style
        self._stop_pulse_state = not self._stop_pulse_state
        style = (
            self._stop_pulse_style
            if self._stop_pulse_state
            else self._stop_base_style
        )
        self._btn_stop.setStyleSheet(style)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _update_run_button_tooltips(self) -> None:
        if not self._run_buttons:
            return

        is_dry = self._is_dry_mode
        active = self._active_job_kind

        def _tip(kind: str) -> str:
            if active == kind:
                if not is_dry:
                    return {
                        "train": "Running training…",
                        "tune": "Running tuning…",
                        "infer": "Running inference…",
                        "xfer": "Running transfer matrix…",
                    }.get(kind, "Running…")
                else:
                    return {
                        "train": "Dry-run – computing planned training steps…",
                        "tune": "Dry-run – computing planned tuning workflow…",
                        "infer": "Dry-run – computing planned inference workflow…",
                        "xfer": "Dry-run – computing planned transfer workflow…",
                    }.get(kind, "Dry-run – computing planned workflow…")
            else:
                if not is_dry:
                    return {
                        "train": "Run training",
                        "tune": "Run tuning",
                        "infer": "Run inference",
                        "xfer": "Run transfer matrix",
                    }.get(kind, "Run workflow")
                else:
                    return {
                        "train": "Run dry (show planned training workflow)",
                        "tune": "Run dry (show planned tuning workflow)",
                        "infer": "Run dry (show planned inference workflow)",
                        "xfer": "Run dry (show planned transfer matrix workflow)",
                    }.get(kind, "Run dry (show planned workflow)")

        for kind, btn in self._run_buttons.items():
            if btn is not None:
                btn.setToolTip(_tip(kind))

    def _update_run_button_icons(self) -> None:
        if not self._run_buttons:
            return
        if self._make_play_icon is None:
            return

        try:
            icon_normal = self._make_play_icon(hollow=False)
            icon_dry = self._make_play_icon(hollow=True)
        except Exception:
            # Never crash just for an icon
            return

        icon = icon_dry if self._is_dry_mode else icon_normal
        for btn in self._run_buttons.values():
            if btn is not None:
                btn.setIcon(icon)

    @staticmethod
    def _build_pulse_style(base: str) -> str:
        """
        Construct a slightly lighter red style on top of the base.
        If no base is provided, fall back to a simple red style.
        """
        if not base:
            base = (
                "QPushButton {background-color: #c62828; color: white;}"
            )
        # Very simple override; you can refine if needed
        pulse = (
            "QPushButton {background-color: #e53935; color: white;}"
        )
        return base + "\n" + pulse
