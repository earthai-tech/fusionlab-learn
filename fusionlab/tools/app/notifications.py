# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: L. Kouadio <etanoyau@gmail.com>

""" App notifications"""

from __future__ import annotations 
import psutil

from PyQt5.QtCore    import ( 
    Qt, 
    QPropertyAnimation, 
    QEasingCurve, 
    QTimer
)
from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtWidgets import (
    QVBoxLayout, 
    QFrame, 
    QLabel,
    QGraphicsOpacityEffect
)

from .utils import _detect_gpu 
 

__all__= ["ToastNotification"]

class ToastNotification(QFrame):
    """A temporary, fading pop-up widget for non-blocking feedback.

    This class creates a self-contained, frameless window that
    displays a message in the center of a parent widget. It is
    designed to appear briefly and then automatically fade out of
    view, providing users with theme-aware status updates without
    interrupting their workflow.

    Parameters
    ----------
    message : str
        The text message to display in the notification.
    parent : QWidget, optional
        The parent widget, used for centering the notification.
        Default is None.
    theme : {'light', 'dark'}, default='light'
        The visual theme to apply, which determines the background
        and border colors of the notification box.
    """
    def __init__(self, message: str, parent=None, theme: str = 'light'):
        super().__init__(parent)
        self.setWindowFlags(
            Qt.FramelessWindowHint | Qt.Tool | Qt.WindowStaysOnTopHint
        )
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_DeleteOnClose)

        # Create the label that will have the visible style
        self.label = QLabel(message, self)
        
        # --- Theme-Aware Styling for the Label ---
        if theme == 'dark':
            bg_color = "rgba(242, 134, 32, 0.9)"  # Orange with opacity
            border_color = "rgba(255, 165, 0, 0.95)"
        else: # Light theme
            bg_color = "rgba(46, 49, 145, 0.9)" # Blue with opacity
            border_color = "rgba(67, 56, 202, 0.95)"

        self.label.setStyleSheet(f"""
            QLabel {{
                background-color: {bg_color};
                color: white;
                font-size: 16px;
                font-weight: bold;
                padding: 15px 25px;
                border-radius: 18px;
                border: 1px solid {border_color};
            }}
        """)
        
        # Main layout for the QFrame container
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0,0,0,0)
        layout.addWidget(self.label)
        
        self.adjustSize()
        self.center_on_parent()

        # --- Set up the Fade-Out Animation ---
        self.opacity_effect = QGraphicsOpacityEffect(self)
        self.setGraphicsEffect(self.opacity_effect)
        
        self.animation = QPropertyAnimation(self.opacity_effect, b"opacity")
        self.animation.setDuration(500)
        self.animation.setStartValue(1.0)
        self.animation.setEndValue(0.0)
        self.animation.setEasingCurve(QEasingCurve.InQuad)
        self.animation.finished.connect(self.close)

    def center_on_parent(self):
        if self.parent():
            parent_rect = self.parent().geometry()
            self.move(
                parent_rect.x() + (parent_rect.width() - self.width()) // 2,
                parent_rect.y() + (parent_rect.height() - self.height()) // 2
            )

    def show_toast(self, duration_ms=1500):
        """Shows the toast and schedules it to fade out."""
        self.opacity_effect.setOpacity(1.0)
        self.show()
        QTimer.singleShot(duration_ms, self.animation.start)



def show_resource_warning(parent=None) -> None:
    """
    Pops up a non-blocking information box when the tuner dialog opens.
    It simply *advises* the user – no runtime blocking.
    """
    total_gb = psutil.virtual_memory().total / 1_073_741_824  # GiB
    gpu_str  = _detect_gpu()

    msg  = []
    msg += ["<b>Hyper-parameter tuning is resource intensive.</b>"]
    msg += [f"Detected system RAM: <b>{total_gb:,.1f} GiB</b>"]
    if gpu_str:
        msg += [f"Detected GPU: <b>{gpu_str}</b> &nbsp;<span "
                "style='color:#2E7D32'>(recommended)</span>"]
    else:
        msg += ["<span style='color:#C62828'>No compatible GPU detected.</span> "
                "Tuning will run on CPU - it can be slow."]

    if total_gb < 8:
        msg += ["<br><br><b style='color:#C62828'>Warning:</b> less than 8 GiB "
                "of RAM may cause out-of-memory errors. Consider reducing "
                "batch-size / trials, or upgrading hardware."]

    # non-modal “information” popup
    box = QMessageBox(parent)
    box.setIcon(QMessageBox.Information)
    box.setWindowTitle("Environment Check")
    box.setText("<br>".join(msg))
    box.setStandardButtons(QMessageBox.Ok)
    box.setModal(False)
    box.show()            # modeless → does not block the dialog

