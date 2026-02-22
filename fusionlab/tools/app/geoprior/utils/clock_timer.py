# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>
#
# Small, reusable "run timer" widget for the GeoPrior GUI.
#
# The widget displays an elapsed time in a watch-like format
#    [ HH:MM:SS:CC ]
# where CC are centiseconds.  It is intended to sit next to the
# "Running / Idle" status label on the main window.
#
# API (core)
# ----------
# timer = RunClockTimer(parent)
# timer.start()        # start / resume counting
# timer.stop()         # pause (keeps elapsed time)
# timer.reset()        # reset to 00:00:00:00 (idle)
# timer.restart()      # reset + start
#
# Extra helpers for the GUI
# -------------------------
# timer.schedule_hibernate(ms=60_000)   # dim after inactivity
# timer.cancel_hibernate()              # cancel dimming
# timer.wake()                          # force full-opacity
#
# Hibernation is *visual only*: the widget fades to a low-opacity
# "sleep" state instead of fully hiding, so it can still respond
# to hover.  The main GUI decides when to call the hibernate
# helpers (typically when a run finishes).

from __future__ import annotations

from typing import Optional

from PyQt5.QtCore import (
    Qt,
    QTimer,
    pyqtSignal,
    QEasingCurve,
    QPropertyAnimation,
)
from PyQt5.QtWidgets import (
    QFrame,
    QLabel,
    QHBoxLayout,
    QSizePolicy,
    QGraphicsOpacityEffect,
)
from PyQt5.QtGui import QFont


class RunClockTimer(QFrame):
    """
    Compact digital run timer with a soft, pill-style appearance.

    The timer uses a QTimer with a centisecond resolution (10 ms) and
    displays elapsed wall-clock time in the format ``[ HH:MM:SS:CC ]``,
    where ``CC`` are centiseconds.

    It is designed to sit next to the global "Running / Idle" label and
    to visually communicate three states:

    * **hidden**: before any run has ever started (handled by the GUI),
    * **active**: job running (accent-colored, full opacity),
    * **hibernating**: job finished, timer dimmed but still readable;
      hovering wakes it back up.

    Signals
    -------
    started : pyqtSignal()
        Emitted when the timer transitions from stopped to running.
    stopped : pyqtSignal()
        Emitted when the timer transitions from running to stopped.
    reset_signal : pyqtSignal()
        Emitted when :meth:`reset` is called.
    elapsed_changed : pyqtSignal(int)
        Emitted on every tick with the elapsed time in milliseconds.
    """

    started = pyqtSignal()
    stopped = pyqtSignal()
    reset_signal = pyqtSignal()
    elapsed_changed = pyqtSignal(int)

    def __init__(
        self,
        parent: Optional[QFrame] = None,
        *,
        interval_ms: int = 10,
        show_centiseconds: bool = True,
        show_brackets: bool = True,
        active_opacity: float = 1.0,
        hibernate_opacity: float = 0.15,
        default_hibernate_ms: int = 60_000,
    ) -> None:

        super().__init__(parent)

        self._elapsed_ms: int = 0
        self._running: bool = False

        self._show_centiseconds = bool(show_centiseconds)
        self._show_brackets = bool(show_brackets)

        self._active_opacity = float(active_opacity)
        self._hibernate_opacity = float(hibernate_opacity)
        self._hibernate_timeout_ms = int(default_hibernate_ms)

        self._hibernating: bool = False

        # Core time-keeping timer
        self._timer = QTimer(self)
        self._timer.setInterval(interval_ms)
        self._timer.timeout.connect(self._on_tick)

        # Main label with monospaced, digital-like font
        self._label = QLabel("00:00:00:00", self)
        self._label.setAlignment(Qt.AlignCenter)

        font = QFont("JetBrains Mono")
        font.setStyleHint(QFont.Monospace)
        font.setPointSize(9)
        font.setBold(True)
        self._label.setFont(font)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 3, 10, 3)
        layout.addWidget(self._label)

        self.setObjectName("runTimer")
        self.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)

        # State-aware style (running vs idle) and chip-like look
        self._apply_styles()

        # Opacity effect + animation for fade-in / fade-out / hibernate
        self._opacity_effect = QGraphicsOpacityEffect(self)
        self.setGraphicsEffect(self._opacity_effect)
        self._opacity_effect.setOpacity(self._active_opacity)

        self._fade_anim = QPropertyAnimation(
            self._opacity_effect, b"opacity", self
        )
        self._fade_anim.setDuration(250)
        self._fade_anim.setEasingCurve(QEasingCurve.InOutCubic)

        # Timer used only for "hibernate after X ms" behaviour
        self._hibernate_timer = QTimer(self)
        self._hibernate_timer.setSingleShot(True)
        self._hibernate_timer.timeout.connect(self._on_hibernate_timeout)

        # Dynamic property so styles can react to running/idle
        # in the stylesheet (see _apply_styles).
        self.setProperty("running", False)

        # Initial display
        self._update_display()

    # ------------------------------------------------------------------ #
    # Styling                                                            #
    # ------------------------------------------------------------------ #
    def _apply_styles(self) -> None:
        """
        Apply a modern chip-style appearance:

        - light gradient capsule background,
        - subtle border,
        - accent-colored digits when running,
        - muted grey-blue when idle.
        """
        self.setStyleSheet(
            """
            QFrame#runTimer {
                /* Soft, light capsule that works on light + mid backgrounds */
                background: qlineargradient(
                    x1:0, y1:0, x2:0, y2:1,
                    stop:0 #f9fafb,
                    stop:1 #edf2f7
                );
                border-radius: 10px;
                border: 1px solid #cbd5e1;
                padding: 2px 8px;
            }

            /* Running state: slightly tinted background + accent digits */
            QFrame#runTimer[running="true"] {
                background: qlineargradient(
                    x1:0, y1:0, x2:0, y2:1,
                    stop:0 #e6fffb,
                    stop:1 #d1fae5
                );
                border-color: #22c55e;
            }
            QFrame#runTimer[running="true"] QLabel {
                color: #047857;  /* teal/green accent */
                font-family: "JetBrains Mono", "Consolas", "Courier New",
                             monospace;
                font-weight: 700;
                letter-spacing: 1px;
            }

            /* Idle / hibernating state: neutral grey-blue digits */
            QFrame#runTimer[running="false"] QLabel {
                color: #64748b;
                font-family: "JetBrains Mono", "Consolas", "Courier New",
                             monospace;
                font-weight: 600;
                letter-spacing: 1px;
            }
            """
        )

    # ------------------------------------------------------------------ #
    # Core timer logic                                                   #
    # ------------------------------------------------------------------ #
    def _on_tick(self) -> None:
        """Internal QTimer callback."""
        if not self._running:
            return
        self._elapsed_ms += self._timer.interval()
        self._update_display()
        self.elapsed_changed.emit(self._elapsed_ms)

    def _update_display(self) -> None:
        """Update the text shown on the label from elapsed ms."""
        ms = max(self._elapsed_ms, 0)
        total_seconds = ms // 1000
        centi = (ms % 1000) // 10

        seconds = total_seconds % 60
        minutes = (total_seconds // 60) % 60
        hours = total_seconds // 3600

        if self._show_centiseconds:
            core = f"{hours:02d}:{minutes:02d}:{seconds:02d}:{centi:02d}"
        else:
            core = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

        if self._show_brackets:
            text = f"[ {core} ]"
        else:
            text = core

        self._label.setText(text)

    # ------------------------------------------------------------------ #
    # Public API: time control                                           #
    # ------------------------------------------------------------------ #
    def start(self) -> None:
        """Start or resume the timer."""
        if self._running:
            return
        self._running = True
        self.setProperty("running", True)
        self.style().unpolish(self)
        self.style().polish(self)

        if not self._timer.isActive():
            self._timer.start()

        # When we (re)start, we should be fully visible
        self.cancel_hibernate()
        self._fade_to(self._active_opacity, duration_ms=160)

        self.started.emit()

    def stop(self) -> None:
        """Pause the timer (elapsed time is preserved)."""
        if not self._running:
            return
        self._running = False
        self.setProperty("running", False)
        self.style().unpolish(self)
        self.style().polish(self)

        # No need to tick if we're idle
        if self._timer.isActive():
            self._timer.stop()

        self.stopped.emit()

    def reset(self) -> None:
        """Reset the timer to zero without starting it."""
        self._elapsed_ms = 0
        self._update_display()
        self.reset_signal.emit()

    def restart(self) -> None:
        """Reset the timer and start immediately."""
        self.reset()
        self.start()

    def is_running(self) -> bool:
        """Return True if the timer is currently running."""
        return self._running

    def elapsed_ms(self) -> int:
        """Return the elapsed time in milliseconds."""
        return self._elapsed_ms

    # ------------------------------------------------------------------ #
    # Hibernation helpers (visual only)                                  #
    # ------------------------------------------------------------------ #
    def schedule_hibernate(self, timeout_ms: Optional[int] = None) -> None:
        """
        Schedule a transition to a dim "hibernating" state.

        The GUI should call this after a job finishes. The timer keeps
        showing the last elapsed time but fades to a low opacity so
        that the log area visually dominates.

        Parameters
        ----------
        timeout_ms : int, optional
            Override the default timeout for this call. If omitted,
            :attr:`_hibernate_timeout_ms` is used (default 60 s).
        """
        if timeout_ms is not None:
            self._hibernate_timeout_ms = int(timeout_ms)
        if self._running:
            # Do not hibernate while a job is active.
            return
        self._hibernate_timer.start(self._hibernate_timeout_ms)

    def cancel_hibernate(self) -> None:
        """
        Cancel any scheduled hibernation and return to full opacity.
        """
        self._hibernate_timer.stop()
        self._hibernating = False
        self._fade_to(self._active_opacity, duration_ms=140)

    def _on_hibernate_timeout(self) -> None:
        """
        Internal slot: fade to low opacity when hibernate kicks in.
        """
        if self._running:
            return
        self._hibernating = True
        self._fade_to(self._hibernate_opacity, duration_ms=420)

    def wake(self) -> None:
        """
        Wake from hibernation: fade back to full opacity.

        This is intended to be used from the main GUI when the user
        hovers the timer area, or when a new job is about to start.
        """
        self._hibernating = False
        self.cancel_hibernate()

    def _fade_to(self, target_opacity: float, *, duration_ms: int) -> None:
        """
        Smoothly animate opacity to `target_opacity`.
        """
        self._fade_anim.stop()
        self._fade_anim.setDuration(max(0, int(duration_ms)))
        self._fade_anim.setStartValue(self._opacity_effect.opacity())
        self._fade_anim.setEndValue(float(target_opacity))
        self._fade_anim.start()

    # ------------------------------------------------------------------ #
    # Hover behaviour (nice for hibernation UX)                          #
    # ------------------------------------------------------------------ #
    def enterEvent(self, event) -> None:  # type: ignore[override]
        """
        When the mouse hovers the timer, gently wake it up if it is
        hibernating (dimmed).
        """
        if not self._running and self._hibernating:
            self.wake()
        super().enterEvent(event)

    def leaveEvent(self, event) -> None:  # type: ignore[override]
        """
        When the mouse leaves and no job is running, (re)schedule
        hibernation so the timer slowly fades away again.
        """
        if not self._running:
            # Use the current configured hibernate delay.
            self.schedule_hibernate()
        super().leaveEvent(event)
