# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>

"""
Reusable popup progress dialog for GeoPrior (and other apps).

Usage
-----
dlg = PopProgressDialog(parent, title="Loading", text="Please wait…")
dlg.show()
cb = dlg.as_fraction_callback()   # f(frac, msg) -> None

# In a long-running function:
cb(0.1, "Step 1…")
cb(0.5, "Halfway…")
cb(1.0, "Done.")

if dlg.was_canceled():
    # handle cancellation (if computation supports it)
"""

from __future__ import annotations

from typing import Optional, Callable

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QLabel,
    QProgressBar,
    QHBoxLayout,
    QPushButton,
    QApplication,
)

from .styles import SECONDARY_TBLUE, PRIMARY, SECONDARY


class PopProgressDialog(QDialog):
    """
    Small, centered popup with a label + progress bar + optional Cancel button.

    - Supports absolute progress via :meth:`set_value`.
    - Supports fractional progress [0, 1] via :meth:`update_fraction`
      or :meth:`as_fraction_callback`.
    """

    def __init__(
        self,
        parent=None,
        *,
        title: str = "Working…",
        text: str = "Please wait…",
        minimum: int = 0,
        maximum: int = 100,
        cancel_text: str = "Cancel",
        cancelable: bool = True,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setWindowModality(Qt.ApplicationModal)
        self.setMinimumWidth(420)
        self._cancelled: bool = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(12)

        self.label = QLabel(text)
        self.label.setWordWrap(True)
        layout.addWidget(self.label)

        self.bar = QProgressBar()
        self.bar.setMinimum(minimum)
        self.bar.setMaximum(maximum)
        self.bar.setTextVisible(True)
        layout.addWidget(self.bar)

        btn_row = QHBoxLayout()
        btn_row.addStretch(1)
        if cancelable:
            self.btn_cancel = QPushButton(cancel_text)
            self.btn_cancel.clicked.connect(self._on_cancel)
            btn_row.addWidget(self.btn_cancel)
        else:
            self.btn_cancel = None
        layout.addLayout(btn_row)

        self._apply_styles()

        # Remove "?" help button
        self.setWindowFlags(
            self.windowFlags() & ~Qt.WindowContextHelpButtonHint
        )

    # ------------------------------------------------------------------
    # Styling
    # ------------------------------------------------------------------
    def _apply_styles(self) -> None:
        self.bar.setStyleSheet(
            """
            QProgressBar {
                border: 1px solid #e5e7eb;
                border-radius: 6px;
                background-color: #f9fafb;
                padding: 1px;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: %s;
                border-radius: 6px;
            }
            """
            % SECONDARY_TBLUE
        )

        if self.btn_cancel is not None:
            self.btn_cancel.setStyleSheet(
                """
                QPushButton {
                    padding: 4px 12px;
                    border-radius: 6px;
                    background-color: %s;
                    color: white;              /* make text visible */
                    font-weight: 500;
                }
                QPushButton:hover {
                    background-color: %s; /* #2563eb slightly darker blue */
                }
                QPushButton:pressed {
                    background-color: #1d4ed8;
                }
                """
                % (PRIMARY, SECONDARY)
            )

    # ------------------------------------------------------------------
    # API
    # ------------------------------------------------------------------
    def _on_cancel(self) -> None:
        self._cancelled = True

    def was_canceled(self) -> bool:
        return self._cancelled

    def set_range(self, minimum: int, maximum: int) -> None:
        self.bar.setMinimum(minimum)
        self.bar.setMaximum(maximum)

    def set_value(
        self,
        value: int,
        message: Optional[str] = None,
    ) -> None:
        """Set absolute progress."""
        if self._cancelled:
            return
        if message is not None:
            self.label.setText(message)
        self.bar.setValue(value)
        QApplication.processEvents()

    def update_fraction(
        self,
        fraction: float,
        message: Optional[str] = None,
    ) -> None:
        """Update progress from a fraction in [0, 1]."""
        if self._cancelled:
            return
        try:
            frac = max(0.0, min(1.0, float(fraction)))
        except Exception:
            frac = 0.0
        minimum = self.bar.minimum()
        maximum = self.bar.maximum()
        val = minimum + int(round(frac * (maximum - minimum)))
        self.set_value(val, message)

    def as_fraction_callback(self) -> Callable[[float, str], None]:
        """
        Return a callback suitable for ``progress_hook`` in read_data:

        >>> cb = dlg.as_fraction_callback()
        >>> read_data("file.csv", progress_hook=cb)
        """

        def cb(frac: float, msg: str) -> None:
            self.update_fraction(frac, msg)

        return cb

    def finish(self) -> None:
        """Set to 100% and close."""
        try:
            self.bar.setValue(self.bar.maximum())
        except Exception:
            pass
        QApplication.processEvents()
        self.close()
