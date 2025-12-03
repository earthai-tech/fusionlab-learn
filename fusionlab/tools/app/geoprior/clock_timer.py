# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>
#
# Small, reusable "run timer" widget for the GeoPrior GUI.
#
# The widget displays an elapsed time in a watch-like format
#    HH:MM:SS:CC
# where CC are centiseconds.  It is intended to sit next to the
# "Running / Idle" status label on the main window.
#
# API
# ----
# timer = RunClockTimer(parent)
# timer.start()      # start / resume counting
# timer.stop()       # pause
# timer.reset()      # reset to 00:00:00:00
# timer.restart()    # reset + start
#
# The look is controlled with an internal stylesheet: black background
# and bright green monospaced digits.

from __future__ import annotations

from typing import Optional

from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtWidgets import QFrame, QLabel, QHBoxLayout, QSizePolicy
from PyQt5.QtGui import QFont


class RunClockTimer(QFrame):
    """
    Digital run timer with a watch-like appearance.

    The timer uses a QTimer with a centisecond resolution (10 ms) and
    displays elapsed wall-clock time in the format ``HH:MM:SS:CC``,
    where ``CC`` are centiseconds.

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
    ) -> None:
        """
        Parameters
        ----------
        parent : QWidget, optional
            Parent widget.
        interval_ms : int, default=10
            Update interval in milliseconds. 10 ms gives a
            centisecond display (HH:MM:SS:CC).
        """
        super().__init__(parent)

        self._elapsed_ms: int = 0
        self._running: bool = False

        self._timer = QTimer(self)
        self._timer.setInterval(interval_ms)
        self._timer.timeout.connect(self._on_tick)

        self._label = QLabel("00:00:00:00", self)
        self._label.setAlignment(Qt.AlignCenter)

        # Monospaced, slightly bold font for a "watch" look
        font = QFont("Consolas")
        font.setStyleHint(QFont.Monospace)
        font.setPointSize(10)
        font.setBold(True)
        self._label.setFont(font)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 2, 6, 2)
        layout.addWidget(self._label)

        self.setObjectName("runTimer")
        self._apply_styles()

        self.setSizePolicy(
            QSizePolicy.Minimum,
            QSizePolicy.Fixed,
        )

    # ------------------------------------------------------------------ #
    # Styling                                                            #
    # ------------------------------------------------------------------ #
    def _apply_styles(self) -> None:
        """
        Apply a dark digital style: black background, green digits.
        """
        self.setStyleSheet(
            """
            QFrame#runTimer {
                background-color: #000000;
                border-radius: 6px;
                padding: 2px 6px;
            }
            QFrame#runTimer QLabel {
                color: #00ff66;
                font-family: "Consolas", "Courier New", monospace;
                font-weight: 600;
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

        self._label.setText(
            f"{hours:02d}:{minutes:02d}:{seconds:02d}:{centi:02d}"
        )

    # ------------------------------------------------------------------ #
    # Public API                                                         #
    # ------------------------------------------------------------------ #
    def start(self) -> None:
        """Start or resume the timer."""
        if self._running:
            return
        self._running = True
        if not self._timer.isActive():
            self._timer.start()
        self.started.emit()

    def stop(self) -> None:
        """Pause the timer (elapsed time is preserved)."""
        if not self._running:
            return
        self._running = False
        # Keep QTimer running so resolution stays consistent; or stop it:
        # self._timer.stop()
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
