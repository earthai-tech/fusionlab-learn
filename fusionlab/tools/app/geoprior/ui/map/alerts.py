# geoprior/ui/map/alerts.py
# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio

"""
geoprior.ui.map.alerts

Widget for the Early Warning System (EWS).
Displays status traffic lights and action buttons.
"""

from __future__ import annotations

from typing import Optional

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QComboBox,
    QCheckBox,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QStyle,
    QWidget,
)

from ...config.store import GeoConfigStore
from .keys import (
    K_ALERT_ENABLED,
    K_ALERT_TRIGGER,
)


class AlertGroup(QFrame):
    """
    Control card for Policy Alerts.
    
    Signals
    -------
    focus_requested : void
        User wants to zoom to critical zones.
    report_requested : void
        User wants to generate a warning report.
    """
    
    focus_requested = pyqtSignal()
    report_requested = pyqtSignal()

    def __init__(
        self,
        *,
        store: GeoConfigStore,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.store = store
        self.setObjectName("mapPanelCard")

        self._build_ui()
        self._connect_store()
        self._update_ui_state()

    def _build_ui(self) -> None:
        # --- Header ---
        root = QFormLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setVerticalSpacing(8)

        # We create the 'card' look manually here
        # or rely on parent adding us to a scroll area.
        # Here we assume we are just the form body.

        self.chk_active = QCheckBox(
            "Enable monitoring",
            self,
        )
        self.chk_active.setToolTip(
            "Scan hotspots for policy breaches."
        )

        self.cmb_trigger = QComboBox(self)
        self.cmb_trigger.addItems([
            "Severity: Critical",
            "Severity: High+",
            "Severity: Any",
            "Manual Threshold",
        ])
        self.cmb_trigger.setToolTip(
            "Condition to trigger alert."
        )

        # --- Traffic Light Status ---
        self.frame_stat = QFrame(self)
        self.frame_stat.setFixedHeight(34)
        
        # Styles > 62 chars broken down
        self._style_ok = (
            "QFrame { "
            "background-color: #2e7d32; "
            "border-radius: 4px; "
            "border: 1px solid #1b5e20; } "
            "QLabel { color: white; "
            "font-weight: bold; }"
        )
        self._style_warn = (
            "QFrame { "
            "background-color: #c62828; "
            "border-radius: 4px; "
            "border: 1px solid #b71c1c; } "
            "QLabel { color: white; "
            "font-weight: bold; }"
        )
        
        self.frame_stat.setStyleSheet(self._style_ok)

        sl = QHBoxLayout(self.frame_stat)
        sl.setContentsMargins(10, 0, 10, 0)

        self.lb_icon = QLabel("✓", self.frame_stat)
        self.lb_msg = QLabel("NORMAL", self.frame_stat)
        self.lb_msg.setAlignment(Qt.AlignCenter)

        sl.addWidget(self.lb_icon)
        sl.addWidget(self.lb_msg, 1)

        # --- Action Buttons ---
        acts = QWidget(self)
        al = QHBoxLayout(acts)
        al.setContentsMargins(0, 0, 0, 0)
        al.setSpacing(6)

        self.btn_focus = QPushButton("Focus", acts)
        self.btn_focus.setToolTip(
            "Zoom to critical zones"
        )
        
        self.btn_rep = QPushButton("Warn", acts)
        self.btn_rep.setToolTip("Issue Warning Report")
        self.btn_rep.setStyleSheet(
            "color: #d32f2f; font-weight: bold;"
        )

        al.addWidget(self.btn_focus)
        al.addWidget(self.btn_rep)

        # --- Layout ---
        root.addRow(self.chk_active)
        root.addRow("Trigger", self.cmb_trigger)
        root.addRow("Status", self.frame_stat)
        root.addRow(acts)

        # --- Local Signals ---
        self.chk_active.toggled.connect(
            self._on_active_toggled
        )
        self.cmb_trigger.currentTextChanged.connect(
            self._on_trigger_changed
        )
        self.btn_focus.clicked.connect(
            self.focus_requested.emit
        )
        self.btn_rep.clicked.connect(
            self.report_requested.emit
        )

    def _connect_store(self) -> None:
        # Initialize from store defaults
        en = bool(self.store.get(K_ALERT_ENABLED, False))
        tr = str(self.store.get(
            K_ALERT_TRIGGER,
            "Severity: Critical",
        ))

        self.chk_active.setChecked(en)
        self._set_combo(self.cmb_trigger, tr)

    def _set_combo(self, cb: QComboBox, txt: str) -> None:
        idx = cb.findText(txt)
        if idx >= 0:
            cb.setCurrentIndex(idx)

    def _on_active_toggled(self, on: bool) -> None:
        self.store.set(K_ALERT_ENABLED, bool(on))
        self._update_ui_state()

    def _on_trigger_changed(self, txt: str) -> None:
        self.store.set(K_ALERT_TRIGGER, str(txt))

    def _update_ui_state(self) -> None:
        on = self.chk_active.isChecked()
        self.cmb_trigger.setEnabled(on)
        self.frame_stat.setVisible(on)
        self.btn_focus.setEnabled(on)
        self.btn_rep.setEnabled(on)

    def sync(self) -> None:
        """Manual refresh from store (for Reset action)."""
        self._connect_store()
        self._update_ui_state()
        
    def set_status(self, warn: bool, count: int = 0) -> None:
        """
        Public API to update the traffic light.
        """
        if not self.chk_active.isChecked():
            return

        if warn:
            self.frame_stat.setStyleSheet(self._style_warn)
            self.lb_icon.setText("⚠")
            self.lb_msg.setText(f"ALERT: {count} ZONES")
        else:
            self.frame_stat.setStyleSheet(self._style_ok)
            self.lb_icon.setText("✓")
            self.lb_msg.setText("NORMAL")