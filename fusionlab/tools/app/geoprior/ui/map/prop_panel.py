# geoprior/ui/map/prop_panel.py
# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>

from __future__ import annotations
from typing import Optional, List

from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QFrame,
    QToolButton,
    QSlider,
    QLabel,
    QCheckBox,
    QSpinBox,
    QStyle,
)
from ...config.store import GeoConfigStore
from .keys import (
    K_PROP_ENABLED,
    K_PROP_YEARS,
    K_PROP_SPEED,
    K_PROP_LOOP,
    K_PROP_VECTORS,
)

class PropagationPanel(QFrame):
    """
    Animation controller for Subsidence Propagation.
    """

    frame_changed = pyqtSignal(int)
    simulation_requested = pyqtSignal(int)

    def __init__(
        self,
        store: GeoConfigStore,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self.store = store
        self.setObjectName("PropPanel")

        self._years: List[int] = []
        self._playing = False
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._on_tick)

        self._init_ui()
        self._connect_store()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)

        # 1. Simulation Setup Area
        setup_row = QHBoxLayout()
        self.chk_enable = QCheckBox("Enable Simulation")
        
        # Sync initial Horizon with store [UPDATED]
        val_y = self.store.get(K_PROP_YEARS, 5)
        self.sp_years = QSpinBox()
        self.sp_years.setRange(1, 50)
        self.sp_years.setValue(int(val_y))
        self.sp_years.setSuffix(" Years")
        
        self.btn_sim = QToolButton()
        self.btn_sim.setText("Generate Scenario")

        setup_row.addWidget(self.chk_enable)
        setup_row.addStretch()
        setup_row.addWidget(QLabel("Horizon:"))
        setup_row.addWidget(self.sp_years)
        setup_row.addWidget(self.btn_sim)

        # 2. Player Controls
        self.player_frame = QFrame()
        self.player_frame.setEnabled(False)
        p_layout = QHBoxLayout(self.player_frame)
        p_layout.setContentsMargins(0, 0, 0, 0)

        self.btn_play = QToolButton()
        self.btn_play.setIcon(
            self.style().standardIcon(QStyle.SP_MediaPlay)
        )

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setTracking(True)

        self.lb_year = QLabel("----")
        self.lb_year.setStyleSheet(
            "font-weight: bold; font-size: 14px;"
            " min-width: 50px;"
        )

        p_layout.addWidget(self.btn_play)
        p_layout.addWidget(self.slider)
        p_layout.addWidget(self.lb_year)

        # 3. Options
        opt_layout = QHBoxLayout()
        self.chk_vectors = QCheckBox("Show Flow Vectors")
        self.chk_loop = QCheckBox("Loop")
        opt_layout.addWidget(self.chk_vectors)
        opt_layout.addWidget(self.chk_loop)
        opt_layout.addStretch()

        layout.addLayout(setup_row)
        layout.addWidget(self.player_frame)
        layout.addLayout(opt_layout)

        # Styles
        self.setFrameShape(QFrame.StyledPanel)
        self.setStyleSheet(
            """
            QFrame#PropPanel {
                background: #fdfdfd; 
                border-top: 1px solid #ddd;
            }
            """
        )
            
        self.chk_enable.setChecked(
            bool(self.store.get(K_PROP_ENABLED, False))
        )
        self.chk_vectors.setChecked(
            bool(self.store.get(K_PROP_VECTORS, True))
        )
        self.chk_loop.setChecked(
            bool(self.store.get(K_PROP_LOOP, False))
        )
        self.player_frame.setVisible(
            bool(self.store.get(K_PROP_ENABLED, False))
        )


    def _connect_store(self):
        self.btn_sim.clicked.connect(self._request_simulation)
        self.btn_play.clicked.connect(self._toggle_play)
        self.slider.valueChanged.connect(self._on_seek)
        self.chk_enable.toggled.connect(
            self._on_enable_toggled
        )
        
        # Save horizon to store when changed [UPDATED]
        self.sp_years.valueChanged.connect(
            lambda v: self.store.set(
                K_PROP_YEARS, int(v)
            )
        )
        
        self.chk_vectors.toggled.connect(
            lambda on: self.store.set(
                K_PROP_VECTORS, bool(on)
            )
        )
        self.chk_loop.toggled.connect(
            lambda on: self.store.set(
                K_PROP_LOOP, bool(on)
            )
        )

    def set_timeline(self, years: List[int]):
        """Call this when data is ready."""
        if not years:
            self._timer.stop()
            self._playing = False
            self._years = []
            self.slider.setRange(0, 0)
            self.slider.setValue(0)
            self.slider.setEnabled(False)
            self.lb_year.setText("----")
            self.player_frame.setEnabled(False)
            return

        self._years = sorted(years)
        self.slider.setEnabled(True)
        self.slider.setRange(0, len(self._years) - 1)
        self.slider.setValue(0)
        self.lb_year.setText(str(self._years[0]))
        self.player_frame.setEnabled(True)

    def _request_simulation(self):
        n = self.sp_years.value()
        self.simulation_requested.emit(n)

    def _toggle_play(self):
        self._playing = not self._playing
        if self._playing:
            self.btn_play.setIcon(
                self.style().standardIcon(
                    QStyle.SP_MediaPause
                )
            )
            # Fetch speed from store (ms)
            ms = self.store.get(K_PROP_SPEED, 800)
            self._timer.start(int(ms))
        else:
            self.btn_play.setIcon(
                self.style().standardIcon(
                    QStyle.SP_MediaPlay
                )
            )
            self._timer.stop()

    def _on_tick(self):
        curr = self.slider.value()
        nxt = curr + 1
        if nxt > self.slider.maximum():
            if self.chk_loop.isChecked():
                nxt = 0
            else:
                self._toggle_play()
                return
        self.slider.setValue(nxt)

    def _on_seek(self, idx):
        if 0 <= idx < len(self._years):
            y = self._years[idx]
            self.lb_year.setText(str(y))
            self.frame_changed.emit(y)

    def _on_enable_toggled(self, on: bool):
        self.store.set(K_PROP_ENABLED, on)
        self.player_frame.setVisible(on)