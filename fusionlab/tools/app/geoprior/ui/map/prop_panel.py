# geoprior/ui/map/prop_panel.py
# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>

from __future__ import annotations
from typing import Optional, List

from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QSignalBlocker
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QFrame,
    QToolButton,
    QColorDialog,
    QSlider,
    QLabel,
    QCheckBox,
    QSpinBox,
    QStyle,
    QComboBox
)
from ...config.store import GeoConfigStore
from .keys import (
    K_PROP_ENABLED,
    K_PROP_YEARS,
    K_PROP_SPEED,
    K_PROP_LOOP,
    K_PROP_VECTORS,
    K_PROP_VECTOR_COLOR_CUSTOM,
    K_PROP_VECTOR_COLOR,
    K_PROP_MODE, 
    K_PROP_LEGEND, 
    MAP_VIEW_HOTSPOTS_ENABLED, 
    
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
        self._loop_anchor: Optional[int] = None
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
        self.btn_sim.setObjectName(
            "btnGenerateScenario"
        )
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
        self.chk_vec_color = QCheckBox("Color")
        self.btn_vec_color = QToolButton()
        self.btn_vec_color.setAutoRaise(True)
        self.btn_vec_color.setFixedSize(22, 22)
        self.btn_vec_color.setVisible(False)
        self.chk_loop = QCheckBox("Loop")
        #  show hotspots during propagation
        self.chk_hot = QCheckBox("Hotspots")
        opt_layout.addWidget(self.chk_vectors)
        opt_layout.addWidget(self.chk_vec_color)
        opt_layout.addWidget(self.btn_vec_color)
        opt_layout.addWidget(self.chk_loop)
        opt_layout.addWidget(self.chk_hot)
        opt_layout.addSpacing(12)

        self.cmb_mode = QComboBox()
        self.cmb_mode.addItem("Absolute", "absolute")
        self.cmb_mode.addItem("Differential", "differential")
        self.cmb_mode.addItem("Risk mask", "risk_mask")

        self.cmb_leg = QComboBox()
        self.cmb_leg.addItem("Global", "global")
        self.cmb_leg.addItem("Frame", "frame")

        self.sp_speed = QSpinBox()
        self.sp_speed.setRange(50, 5000)
        self.sp_speed.setSingleStep(50)
        self.sp_speed.setSuffix(" ms")

        opt_layout.addWidget(QLabel("Mode:"))
        opt_layout.addWidget(self.cmb_mode)
        opt_layout.addSpacing(8)
        opt_layout.addWidget(QLabel("Legend:"))
        opt_layout.addWidget(self.cmb_leg)
        opt_layout.addSpacing(8)
        opt_layout.addWidget(QLabel("Speed:"))
        opt_layout.addWidget(self.sp_speed)
        opt_layout.addStretch(1)

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

        self.chk_vec_color.toggled.connect(
            self._on_vec_color_custom
        )
        self.btn_vec_color.clicked.connect(
            self._pick_vec_color
        )
        self.chk_loop.toggled.connect(
            lambda on: self.store.set(
                K_PROP_LOOP, bool(on)
            )
        )

        self.chk_hot.toggled.connect(
            lambda v: self.store.set(
                MAP_VIEW_HOTSPOTS_ENABLED,
                bool(v),
            )
        )

        self.cmb_mode.currentIndexChanged.connect(
            lambda _i: self.store.set(
                K_PROP_MODE,
                str(self.cmb_mode.currentData() or ""),
            )
        )
        self.cmb_leg.currentIndexChanged.connect(
            lambda _i: self.store.set(
                K_PROP_LEGEND,
                str(self.cmb_leg.currentData() or ""),
            )
        )
        self.sp_speed.valueChanged.connect(
            lambda v: self.store.set(K_PROP_SPEED, int(v))
        )

        # Store → UI synchronization
        self.store.config_changed.connect(
            self._on_store_changed
        )
        self._sync_from_store()

    def _on_store_changed(self, keys) -> None:
        ks = set(keys or [])
        watch = {
            K_PROP_ENABLED,
            K_PROP_YEARS,
            K_PROP_SPEED,
            K_PROP_LOOP,
            K_PROP_VECTORS,
            K_PROP_MODE,
            K_PROP_LEGEND,
            MAP_VIEW_HOTSPOTS_ENABLED,
            K_PROP_VECTOR_COLOR_CUSTOM,
            K_PROP_VECTOR_COLOR,
        }
        if not ks.intersection(watch):
            return
        self._sync_from_store()
    
    def _sync_from_store(self) -> None:
        # Store → UI (block signals to avoid loops)
        if self.store is None:
            return
    
        en = bool(self.store.get(K_PROP_ENABLED, False))
        yrs = int(self.store.get(K_PROP_YEARS, 5) or 5)
        loop = bool(self.store.get(K_PROP_LOOP, False))
        vec = bool(self.store.get(K_PROP_VECTORS, True))
        c_on = bool(
            self.store.get(
                K_PROP_VECTOR_COLOR_CUSTOM,
                False,
            )
        )
        col = str(
            self.store.get(
                K_PROP_VECTOR_COLOR,
                "#d7263d",
            )
            or "#d7263d"
        )
        spd = int(self.store.get(K_PROP_SPEED, 800) or 800)
        mode = str(self.store.get(K_PROP_MODE, "absolute") or "absolute")
        leg = str(self.store.get(K_PROP_LEGEND, "global") or "global")
        hot = bool(self.store.get(MAP_VIEW_HOTSPOTS_ENABLED, False))

        def _set_by_data(cmb: QComboBox, v: str) -> None:
            vv = str(v or "")
            for i in range(cmb.count()):
                if str(cmb.itemData(i) or "") == vv:
                    cmb.setCurrentIndex(i)
                    return
        
        with QSignalBlocker(self.chk_enable):
            self.chk_enable.setChecked(en)
    
        with QSignalBlocker(self.sp_years):
            self.sp_years.setValue(int(yrs))
    
        with QSignalBlocker(self.chk_loop):
            self.chk_loop.setChecked(loop)
    
        with QSignalBlocker(self.chk_vectors):
            self.chk_vectors.setChecked(vec)

        with QSignalBlocker(self.chk_vec_color):
            self.chk_vec_color.setChecked(c_on)
        self.btn_vec_color.setVisible(bool(c_on))
        self._set_vec_color_swatch(col)
            
        with QSignalBlocker(self.chk_hot):
            self.chk_hot.setChecked(hot)

        with QSignalBlocker(self.cmb_mode):
            _set_by_data(self.cmb_mode, mode)

        with QSignalBlocker(self.cmb_leg):
            _set_by_data(self.cmb_leg, leg)

        with QSignalBlocker(self.sp_speed):
            self.sp_speed.setValue(spd)
        self.player_frame.setVisible(en)

    def _set_vec_color_swatch(self, col: str) -> None:
        c = str(col or "#d7263d").strip()
        if not c:
            c = "#d7263d"
        self.btn_vec_color.setStyleSheet(
            "QToolButton {"
            f"background: {c};"
            "border: 1px solid #bbb;"
            "border-radius: 3px;"
            "}"
        )

    def _on_vec_color_custom(self, on: bool) -> None:
        self.store.set(
            K_PROP_VECTOR_COLOR_CUSTOM,
            bool(on),
        )
        self.btn_vec_color.setVisible(bool(on))
        if not on:
            return
        cur = str(
            self.store.get(K_PROP_VECTOR_COLOR, "#d7263d")
            or "#d7263d"
        )
        self._set_vec_color_swatch(cur)

    def _pick_vec_color(self) -> None:
        cur = str(
            self.store.get(K_PROP_VECTOR_COLOR, "#d7263d")
            or "#d7263d"
        )
        qc = QColor(cur)
        if not qc.isValid():
            qc = QColor("#d7263d")
        out = QColorDialog.getColor(qc, self)
        if not out.isValid():
            return
        col = str(out.name())
        self.store.set(K_PROP_VECTOR_COLOR, col)
        self._set_vec_color_swatch(col)


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
            self._loop_anchor = int(self.slider.value())
            self.btn_play.setIcon(
                self.style().standardIcon(
                    QStyle.SP_MediaPause
                )
            )
            # Fetch speed from store (ms)
            ms = self.store.get(K_PROP_SPEED, 800)
            self._timer.start(int(ms))
        else:
            self._loop_anchor = None
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
                a = self._loop_anchor
                if a is None:
                    nxt = 0
                else:
                    nxt = max(0, min(int(a), self.slider.maximum()))
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
        on = bool(on)
    
        if not on:
            self._timer.stop()
            self._playing = False
            self._loop_anchor = None
            self.btn_play.setIcon(
                self.style().standardIcon(QStyle.SP_MediaPlay)
            )
    
        self.store.set(K_PROP_ENABLED, on)
        self.player_frame.setVisible(on)
