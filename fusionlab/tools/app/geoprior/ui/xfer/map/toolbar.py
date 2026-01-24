# geoprior/ui/xfer/map/toolbar.py
# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>

"""
geoprior.ui.xfer.map.toolbar

Modern 3-row toolbar:
1) Context (A/B dataset identity)
2) View controls (segmented pills + value + actions)
3) Time scrubber (video-like)

UI controls only:
- Emits signals
- Holds no file IO / pandas
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMenu,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QStyle,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ...icon_utils import try_icon 

@dataclass
class DatasetChoice:
    """Minimal descriptor for combo population."""
    text: str
    data: Any


class XferMapToolbar(QWidget):
    """
    3-row toolbar with segmented controls.

    The controller owns the store sync.
    """

    request_open_options = pyqtSignal()
    request_fit = pyqtSignal()
    request_refresh = pyqtSignal()
    changed = pyqtSignal()
    request_expand = pyqtSignal(bool)
    request_mode_switch = pyqtSignal(str)

    def __init__(
        self,
        *,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._play_timer = QTimer(self)
        self._play_timer.setInterval(320)
        self._play_timer.timeout.connect(self._on_play_tick)

        self._build_ui()
        self._connect()

    # -------------------------
    # Public API (controller)
    # -------------------------

    def set_expanded(self, on: bool) -> None:
        self.btn_expand.blockSignals(True)
        try:
            self.btn_expand.setChecked(bool(on))
        finally:
            self.btn_expand.blockSignals(False)
        self._update_expand_icon(bool(on))

    def _update_expand_icon(self, on: bool) -> None:
        if bool(on):
            self.btn_expand.setToolTip("Restore layout")
            ico = self.style().standardIcon(
                QStyle.SP_TitleBarNormalButton
            )
        else:
            self.btn_expand.setToolTip("Expand map")
            ico = self.style().standardIcon(
                QStyle.SP_TitleBarMaxButton
            )
        self.btn_expand.setIcon(ico)

    def _on_expand_toggled(self, on: bool) -> None:
        self._update_expand_icon(bool(on))
        self.request_expand.emit(bool(on))

    def set_city_choices(
        self,
        *,
        a: List[DatasetChoice],
        b: List[DatasetChoice],
    ) -> None:
        self._fill_combo(self.cmb_city_a, a)
        self._fill_combo(self.cmb_city_b, b)

    def set_job_choices(
        self,
        *,
        a: List[DatasetChoice],
        b: List[DatasetChoice],
    ) -> None:
        self._fill_combo(self.cmb_job_a, a)
        self._fill_combo(self.cmb_job_b, b)

    def set_file_choices(
        self,
        *,
        a: List[DatasetChoice],
        b: List[DatasetChoice],
    ) -> None:
        self._fill_combo(self.cmb_file_a, a)
        self._fill_combo(self.cmb_file_b, b)

    def set_value_choices(
        self,
        items: List[DatasetChoice],
    ) -> None:
        self._fill_combo(self.cmb_value, items)

    def set_time_range(
        self,
        *,
        step_min: int,
        step_max: int,
        step: int,
    ) -> None:
        mn = int(step_min)
        mx = int(step_max)
        st = int(step)

        mx = max(1, mx)
        st = max(mn, min(st, mx))

        self.sld_step.blockSignals(True)
        try:
            self.sld_step.setRange(mn, mx)
            self.sld_step.setValue(st)
        finally:
            self.sld_step.blockSignals(False)

        self._update_step_label()

    def set_current_data(
        self,
        cmb: QComboBox,
        data: Any,
    ) -> None:
        for i in range(cmb.count()):
            if cmb.itemData(i) == data:
                cmb.blockSignals(True)
                try:
                    cmb.setCurrentIndex(i)
                finally:
                    cmb.blockSignals(False)
                return

    def set_split(self, split: str) -> None:
        self._set_seg(self._seg_split_map, split)

    def set_value(self, value: str) -> None:
        self.set_current_data(self.cmb_value, value)

    def set_overlay(self, overlay: str) -> None:
        self._set_seg(self._seg_ovl_map, overlay)

    def set_time_mode(self, mode: str) -> None:
        self._set_seg(self._seg_time_map, mode)
        self._update_time_prefix()

    def set_shared(self, shared: bool) -> None:
        self._set_seg(self._seg_scale_map, bool(shared))
        
    def set_points_mode(self, mode: str) -> None:
        self._set_seg(self._seg_pts_map, str(mode or "all"))

    def set_marker_shape(self, shape: str) -> None:
        self.set_current_data(self.cmb_shape, str(shape or "auto"))

    def set_marker_size(self, px: int) -> None:
        self.sp_size.blockSignals(True)
        try:
            self.sp_size.setValue(int(px))
        finally:
            self.sp_size.blockSignals(False)

    def set_hotspot_topn(self, n: int) -> None:
        self.sp_topn.blockSignals(True)
        try:
            self.sp_topn.setValue(int(n))
        finally:
            self.sp_topn.blockSignals(False)

    def set_pulse(self, on: bool) -> None:
        self.btn_pulse.blockSignals(True)
        try:
            self.btn_pulse.setChecked(bool(on))
        finally:
            self.btn_pulse.blockSignals(False)

    def set_play_ms(self, ms: int) -> None:
        self.set_current_data(self.cmb_speed, int(ms))
        self._apply_play_ms()

    def get_ui_state(self) -> Dict[str, Any]:
        return {
            "city_a": self.cmb_city_a.currentData(),
            "job_a": self.cmb_job_a.currentData(),
            "file_a": self.cmb_file_a.currentData(),
            "city_b": self.cmb_city_b.currentData(),
            "job_b": self.cmb_job_b.currentData(),
            "file_b": self.cmb_file_b.currentData(),
            "split": self._seg_value(self._seg_split),
            "value": self.cmb_value.currentData(),
            "overlay": self._seg_value(self._seg_ovl),
            "shared": bool(self._seg_value(self._seg_scale)),
            "time_mode": self._seg_value(self._seg_time),
            "step": int(self.sld_step.value()),
            "points_mode": self._seg_value(self._seg_pts),
            "marker_shape": self.cmb_shape.currentData(),
            "marker_size": int(self.sp_size.value()),
            "hotspot_top_n": int(self.sp_topn.value()),
            "pulse": bool(self.btn_pulse.isChecked()),
            "play_ms": int(self.cmb_speed.currentData() or 320),
            "insight": bool(self.btn_insight.isChecked()),
        }


    # -------------------------
    # UI builders
    # -------------------------
    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(6)

        # -------------------------
        # Row 1: Context (A/B)
        # -------------------------
        ctx = QWidget(self)
        g = QGridLayout(ctx)
        g.setContentsMargins(0, 0, 0, 0)
        g.setHorizontalSpacing(8)
        g.setVerticalSpacing(6)

        self.lbl_chip_a = QLabel("A", self)
        self.lbl_chip_a.setObjectName("xferChipA")
        self.lbl_chip_a.setAlignment(Qt.AlignCenter)

        self.lbl_chip_b = QLabel("B", self)
        self.lbl_chip_b.setObjectName("xferChipB")
        self.lbl_chip_b.setAlignment(Qt.AlignCenter)

        self.cmb_city_a = QComboBox(self)
        self.cmb_job_a = QComboBox(self)
        self.cmb_file_a = QComboBox(self)

        self.cmb_city_b = QComboBox(self)
        self.cmb_job_b = QComboBox(self)
        self.cmb_file_b = QComboBox(self)

        self.btn_more = QToolButton(self)
        self.btn_more.setText("⋯")
        self.btn_more.setObjectName("miniAction")
        self.btn_more.setToolTip("More")
        self.btn_more.setPopupMode(
            QToolButton.InstantPopup
        )

        self._more_menu = QMenu(self)
        self._act_mode = self._more_menu.addAction(
            "Go to Run mode"
        )
        self._act_mode.triggered.connect(
            lambda *_: self._on_mode_action()
        )

        self._more_menu.addSeparator()

        act_reset = self._more_menu.addAction(
            "Reset view controls"
        )
        act_reset.triggered.connect(
            self._reset_view_controls
        )

        act_stop = self._more_menu.addAction(
            "Stop playback"
        )
        act_stop.triggered.connect(
            self._stop_playback
        )

        self.btn_more.setMenu(self._more_menu)

        # A row
        g.addWidget(self.lbl_chip_a, 0, 0)
        g.addWidget(self.cmb_city_a, 0, 1)
        g.addWidget(self.cmb_job_a, 0, 2)
        g.addWidget(self.cmb_file_a, 0, 3)
        g.addWidget(self.btn_more, 0, 4)

        # B row
        g.addWidget(self.lbl_chip_b, 1, 0)
        g.addWidget(self.cmb_city_b, 1, 1)
        g.addWidget(self.cmb_job_b, 1, 2)
        g.addWidget(self.cmb_file_b, 1, 3)

        g.setColumnStretch(3, 1)

        root.addWidget(ctx)

        # -------------------------
        # Row 2: View controls
        # -------------------------
        view = QWidget(self)
        gv = QGridLayout(view)
        gv.setContentsMargins(0, 0, 0, 0)
        gv.setHorizontalSpacing(10)
        gv.setVerticalSpacing(6)

        # Segmented: Split
        w_split, self._seg_split, self._seg_split_map = (
            self._make_seg(
                [
                    ("Val", "val"),
                    ("Test", "test"),
                ]
            )
        )
        gv.addWidget(QLabel("Split:"), 0, 0)
        gv.addWidget(w_split, 0, 1)

        # Segmented: Overlay
        w_ovl, self._seg_ovl, self._seg_ovl_map = (
            self._make_seg(
                [
                    ("A", "a"),
                    ("A+B", "both"),
                    ("B", "b"),
                ]
            )
        )
        gv.addWidget(QLabel("Overlay:"), 0, 2)
        gv.addWidget(w_ovl, 0, 3)

        # Segmented: Scale
        w_scale, self._seg_scale, self._seg_scale_map = (
            self._make_seg(
                [
                    ("Shared", True),
                    ("Auto", False),
                ]
            )
        )
        gv.addWidget(QLabel("Scale:"), 0, 4)
        gv.addWidget(w_scale, 0, 5)

        # Segmented: Time mode
        w_time, self._seg_time, self._seg_time_map = (
            self._make_seg(
                [
                    ("Step", "forecast_step"),
                    ("Year", "year"),
                ]
            )
        )
        gv.addWidget(QLabel("Time:"), 0, 6)
        gv.addWidget(w_time, 0, 7)

        # Value dropdown + actions (row 1)
        self.cmb_value = QComboBox(self)
        
        # Keep it compact so action row stays readable.
        self.cmb_value.setMinimumContentsLength(6)
        self.cmb_value.setMinimumWidth(120)
        self.cmb_value.setMaximumWidth(220)
        self.cmb_value.setSizePolicy(
            QSizePolicy.Fixed,
            QSizePolicy.Fixed,
        )
        
        act_row = QHBoxLayout()
        act_row.setContentsMargins(0, 0, 0, 0)
        act_row.setSpacing(8)

        # --- Points mode (All | Hot | Hot+Base)
        w_pts, self._seg_pts, self._seg_pts_map = self._make_seg(
            [
                ("All", "all"),
                ("Hot", "hotspots"),
                ("Hot+", "hotspots_plus"),
            ]
        )

        # --- Shape
        self.cmb_shape = QComboBox(self)
        self.cmb_shape.setMinimumWidth(90)
        self.cmb_shape.setMaximumWidth(120)
        self.cmb_shape.addItem("Auto", "auto")
        self.cmb_shape.addItem("Circle", "circle")
        self.cmb_shape.addItem("Triangle", "triangle")
        self.cmb_shape.addItem("Diamond", "diamond")
        self.cmb_shape.addItem("Square", "square")
        self.cmb_shape.setToolTip(
            "Marker shape.\n"
            "Auto = circle (base), triangle (hotspots)."
        )

        # --- Size (px)
        self.sp_size = QSpinBox(self)
        self.sp_size.setRange(2, 18)
        self.sp_size.setSingleStep(1)
        self.sp_size.setValue(6)
        self.sp_size.setFixedWidth(58)
        self.sp_size.setToolTip("Marker size (px).")

        # --- Hotspots: Top-N
        self.sp_topn = QSpinBox(self)
        self.sp_topn.setRange(1, 50)
        self.sp_topn.setSingleStep(1)
        self.sp_topn.setValue(8)
        self.sp_topn.setFixedWidth(58)
        self.sp_topn.setToolTip("Hotspot count (Top-N).")

        # --- Pulse (hotspots)
        self.btn_pulse = QToolButton(self)
        self.btn_pulse.setObjectName("miniAction")
        self.btn_pulse.setAutoRaise(True)
        self.btn_pulse.setCheckable(True)
        self.btn_pulse.setChecked(True)
        self.btn_pulse.setText("Pulse")
        self.btn_pulse.setToolTip(
            "Pulse hotspots (visual attention)."
        )

        # --- Playback speed (ms/frame)
        self.cmb_speed = QComboBox(self)
        self.cmb_speed.setMinimumWidth(110)
        self.cmb_speed.addItem("Slow", 520)
        self.cmb_speed.addItem("Normal", 320)
        self.cmb_speed.addItem("Fast", 180)
        self.cmb_speed.setCurrentIndex(1)
        self.cmb_speed.setToolTip(
            "Playback speed (ms per step)."
        )
        
        self.btn_insight = QToolButton(self)
        self.btn_insight.setObjectName("miniAction")
        self.btn_insight.setAutoRaise(True)
        self.btn_insight.setCheckable(True)
        self.btn_insight.setToolTip(
            "Transfer insight (A↔B)"
        )
        
        ico = try_icon("xfer_insight.svg")
        if ico is None:
            ico = self.style().standardIcon(
                QStyle.SP_MessageBoxInformation
            )
        self.btn_insight.setIcon(ico)
        

        # --- Refresh
        self.btn_refresh = QToolButton(self)
        self.btn_refresh.setObjectName("miniAction")
        self.btn_refresh.setAutoRaise(True)
        self.btn_refresh.setToolTip("Refresh")
        self.btn_refresh.setIcon(
            self.style().standardIcon(QStyle.SP_BrowserReload)
        )

        # --- Fit (moved to end)
        self.btn_fit = QToolButton(self)
        self.btn_fit.setObjectName("miniAction")
        self.btn_fit.setAutoRaise(True)
        self.btn_fit.setToolTip("Fit to layers")

        ico = try_icon("mapfit_icon.svg")
        if ico is not None:
            self.btn_fit.setIcon(ico)
        else:
            self.btn_fit.setIcon(
                self.style().standardIcon(
                    QStyle.SP_TitleBarMaxButton
                )
            )

        # --- Reset (moved to end)
        self.btn_reset = QToolButton(self)
        self.btn_reset.setObjectName("miniAction")
        self.btn_reset.setAutoRaise(True)
        self.btn_reset.setToolTip("Reset view controls")
        self.btn_reset.setIcon(
            self.style().standardIcon(QStyle.SP_DialogResetButton)
        )
        # --- Expand / Restore (focus map)
        self.btn_expand = QToolButton(self)
        self.btn_expand.setObjectName("miniAction")
        self.btn_expand.setAutoRaise(True)
        self.btn_expand.setCheckable(True)
        self._update_expand_icon(False)

        # Layout order: options first, Fit+Reset last
        act_row.addWidget(QLabel("Pts:"))
        act_row.addWidget(w_pts)

        act_row.addSpacing(6)
        act_row.addWidget(QLabel("Shape:"))
        act_row.addWidget(self.cmb_shape)

        act_row.addWidget(QLabel("Size:"))
        act_row.addWidget(self.sp_size)

        act_row.addWidget(QLabel("Top:"))
        act_row.addWidget(self.sp_topn)

        act_row.addWidget(self.btn_pulse)

        act_row.addSpacing(8)
        act_row.addWidget(QLabel("Speed:"))
        act_row.addWidget(self.cmb_speed)

        act_row.addStretch(1)

        act_row.addWidget(self.btn_insight)
        act_row.addWidget(self.btn_refresh)
        act_row.addWidget(self.btn_fit)
        act_row.addWidget(self.btn_reset)
        act_row.addWidget(self.btn_expand)

        w_act = QWidget(self)
        w_act.setLayout(act_row)
        w_act.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Fixed,
        )

        gv.addWidget(QLabel("Value:"), 1, 0)
        gv.addWidget(self.cmb_value, 1, 1)
        gv.addWidget(w_act, 1, 2, 1, 6)

        gv.setColumnStretch(7, 1)
        root.addWidget(view)

        # -------------------------
        # Row 3: Time scrubber
        # -------------------------
        trow = QWidget(self)
        ht = QHBoxLayout(trow)
        ht.setContentsMargins(0, 0, 0, 0)
        ht.setSpacing(8)

        self.lbl_time_prefix = QLabel("Step", self)
        self.lbl_time_prefix.setObjectName("xferTimePrefix")

        self.btn_prev = QToolButton(self)
        self.btn_prev.setObjectName("miniAction")
        self.btn_prev.setAutoRaise(True)
        self.btn_prev.setToolTip("Previous")
        self.btn_prev.setIcon(
            self.style().standardIcon(
                QStyle.SP_ArrowBack
            )
        )

        self.btn_play = QToolButton(self)
        self.btn_play.setObjectName("miniAction")
        self.btn_play.setAutoRaise(True)
        self.btn_play.setToolTip("Play / Pause")
        self.btn_play.setCheckable(True)
        self.btn_play.setIcon(
            self.style().standardIcon(
                QStyle.SP_MediaPlay
            )
        )

        self.btn_next = QToolButton(self)
        self.btn_next.setObjectName("miniAction")
        self.btn_next.setAutoRaise(True)
        self.btn_next.setToolTip("Next")
        self.btn_next.setIcon(
            self.style().standardIcon(
                QStyle.SP_ArrowForward
            )
        )

        self.sld_step = QSlider(Qt.Horizontal, self)
        self.sld_step.setRange(1, 1)
        self.sld_step.setValue(1)
        self.sld_step.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Fixed,
        )
        self.sld_step.setMinimumWidth(260)

        self.lbl_step = QLabel("1 / 1", self)
        self.lbl_step.setObjectName("xferStepLabel")
        self.lbl_step.setMinimumWidth(72)

        ht.addWidget(self.lbl_time_prefix)
        ht.addWidget(self.btn_prev)
        ht.addWidget(self.btn_play)
        ht.addWidget(self.btn_next)
        ht.addWidget(self.sld_step, 1)
        ht.addWidget(self.lbl_step)

        root.addWidget(trow)

        # -------------------------
        # Tooltips + sizing
        # -------------------------
        # keep city/job/file combos in the loop (exclude cmb_value)
        for cmb in (
            self.cmb_city_a,
            self.cmb_job_a,
            self.cmb_file_a,
            self.cmb_city_b,
            self.cmb_job_b,
            self.cmb_file_b,
        ):
            cmb.setSizeAdjustPolicy(
                QComboBox.AdjustToMinimumContentsLengthWithIcon
            )
            cmb.setMinimumContentsLength(10)
        
        # now value stays compact
        self.cmb_value.setSizeAdjustPolicy(
            QComboBox.AdjustToMinimumContentsLengthWithIcon
        )
        self.cmb_value.setMinimumContentsLength(6)
        self.cmb_value.setMinimumWidth(110)
        self.cmb_value.setMaximumWidth(180)
        
        self.cmb_city_a.setMinimumWidth(120)
        self.cmb_job_a.setMinimumWidth(150)
        self.cmb_file_a.setMinimumWidth(240)

        self.cmb_city_b.setMinimumWidth(120)
        self.cmb_job_b.setMinimumWidth(150)
        self.cmb_file_b.setMinimumWidth(240)

        self.cmb_city_a.setToolTip("Select city for A")
        self.cmb_job_a.setToolTip("Select run/job for A")
        self.cmb_file_a.setToolTip("Select forecast CSV for A")

        self.cmb_city_b.setToolTip("Select city for B")
        self.cmb_job_b.setToolTip("Select run/job for B")
        self.cmb_file_b.setToolTip("Select forecast CSV for B")

        self.cmb_value.setToolTip("Value to visualize")

        self.sld_step.setToolTip("Forecast step / year index")

        # defaults for pills
        self.set_split("val")
        self.set_overlay("both")
        self.set_shared(True)
        self.set_time_mode("forecast_step")

    def _connect(self) -> None:
        self.btn_fit.clicked.connect(self.request_fit)
        self.btn_refresh.clicked.connect(self.request_refresh)

        for w in (
            self.cmb_city_a,
            self.cmb_job_a,
            self.cmb_file_a,
            self.cmb_city_b,
            self.cmb_job_b,
            self.cmb_file_b,
            self.cmb_value,
        ):
            w.currentIndexChanged.connect(self.changed)

        self._seg_split.buttonClicked.connect(self.changed)
        self._seg_ovl.buttonClicked.connect(self.changed)
        self._seg_scale.buttonClicked.connect(self.changed)
        self._seg_time.buttonClicked.connect(self._on_time_mode)
        self._seg_time.buttonClicked.connect(self.changed)

        self.sld_step.valueChanged.connect(self.changed)
        self.sld_step.valueChanged.connect(
            lambda _v: self._update_step_label()
        )

        self.btn_prev.clicked.connect(
            lambda: self._nudge_step(-1)
        )
        self.btn_next.clicked.connect(
            lambda: self._nudge_step(+1)
        )
        self.btn_play.toggled.connect(self._on_play)
        self.btn_expand.toggled.connect(
            self._on_expand_toggled
        )
        
        # --- map render knobs (must trigger controller refresh)
        self._seg_pts.buttonClicked.connect(self.changed)
        
        self.cmb_shape.currentIndexChanged.connect(self.changed)
        self.sp_size.valueChanged.connect(self.changed)
        self.sp_topn.valueChanged.connect(self.changed)
        
        self.btn_pulse.toggled.connect(self.changed)
        
        self.cmb_speed.currentIndexChanged.connect(
            self._on_speed_changed
        )
        self.cmb_speed.currentIndexChanged.connect(self.changed)
        self.btn_insight.toggled.connect(self.changed)
        # Ensure timer matches current UI speed.
        self._apply_play_ms()
        
    def set_mode(self, mode: str) -> None:
        m = str(mode or "").strip().lower()
        if m not in ("run", "map"):
            m = "map"
    
        self._cur_mode = m
        tgt = "run" if m == "map" else "map"
        self._mode_target = tgt
    
        if hasattr(self, "_act_mode"):
            if tgt == "run":
                self._act_mode.setText("Go to Run mode")
            else:
                self._act_mode.setText("Go to Map mode")
    
    
    def _on_mode_action(self) -> None:
        tgt = getattr(self, "_mode_target", "run")
        self.request_mode_switch.emit(str(tgt))
        # legacy: keep existing wiring alive
        self.request_open_options.emit()

    # -------------------------
    # Segmented helpers
    # -------------------------
    def _make_seg(
        self,
        items: List[Tuple[str, Any]],
    ) -> Tuple[QWidget, QButtonGroup, Dict[Any, QToolButton]]:
        w = QWidget(self)
        w.setObjectName("xferSeg")

        row = QHBoxLayout(w)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(0)

        grp = QButtonGroup(w)
        grp.setExclusive(True)

        out: Dict[Any, QToolButton] = {}

        n = len(items)
        for i, (txt, val) in enumerate(items):
            b = QToolButton(w)
            b.setObjectName("xferSegBtn")
            b.setText(str(txt))
            b.setCheckable(True)
            b.setAutoRaise(True)
            b.setProperty("segValue", val)

            if i == 0:
                b.setProperty("pos", "left")
            elif i == n - 1:
                b.setProperty("pos", "right")
            else:
                b.setProperty("pos", "mid")

            grp.addButton(b, i)
            out[val] = b
            row.addWidget(b)

        # pick first by default
        if items:
            out[items[0][1]].setChecked(True)

        return w, grp, out

    def _seg_value(self, grp: QButtonGroup) -> Any:
        b = grp.checkedButton()
        if b is None:
            return None
        return b.property("segValue")

    def _set_seg(
        self,
        mp: Dict[Any, QToolButton],
        value: Any,
    ) -> None:
        b = mp.get(value, None)
        if b is None:
            return

        for btn in mp.values():
            btn.blockSignals(True)
        try:
            b.setChecked(True)
        finally:
            for btn in mp.values():
                btn.blockSignals(False)

    # -------------------------
    # Time scrubber
    # -------------------------
    def _update_time_prefix(self) -> None:
        mode = str(self._seg_value(self._seg_time) or "")
        if mode == "year":
            self.lbl_time_prefix.setText("Year")
        else:
            self.lbl_time_prefix.setText("Step")

    def _update_step_label(self) -> None:
        v = int(self.sld_step.value())
        mx = int(self.sld_step.maximum())
        self.lbl_step.setText(f"{v} / {mx}")

    def _nudge_step(self, delta: int) -> None:
        v = int(self.sld_step.value())
        mn = int(self.sld_step.minimum())
        mx = int(self.sld_step.maximum())
        nv = max(mn, min(v + int(delta), mx))
        if nv != v:
            self.sld_step.setValue(nv)

    def _on_play(self, on: bool) -> None:
        if bool(on):
            self.btn_play.setIcon(
                self.style().standardIcon(
                    QStyle.SP_MediaPause
                )
            )
            self._play_timer.start()
            return

        self.btn_play.setIcon(
            self.style().standardIcon(
                QStyle.SP_MediaPlay
            )
        )
        self._play_timer.stop()

    def _on_play_tick(self) -> None:
        v = int(self.sld_step.value())
        mx = int(self.sld_step.maximum())
        if v >= mx:
            self.btn_play.setChecked(False)
            return
        self.sld_step.setValue(v + 1)

    def _stop_playback(self) -> None:
        if self.btn_play.isChecked():
            self.btn_play.setChecked(False)

    def _on_time_mode(self) -> None:
        self._update_time_prefix()
        
    def _apply_play_ms(self) -> None:
        ms = int(self.cmb_speed.currentData() or 320)
        ms = max(60, min(ms, 1500))
        self._play_timer.setInterval(ms)

    def _on_speed_changed(self) -> None:
        self._apply_play_ms()

    # -------------------------
    # Reset / fill
    # -------------------------
    def _reset_view_controls(self) -> None:
        self._stop_playback()
        self.set_split("val")
        self.set_overlay("both")
        self.set_shared(True)
        self.set_time_mode("forecast_step")
        self.set_value("auto")
        self.btn_insight.setChecked(False)

        self.set_points_mode("all")
        self.set_marker_shape("auto")
        self.set_marker_size(6)
        self.set_hotspot_topn(8)
        self.set_pulse(True)
        self.set_play_ms(320)

        self.sld_step.setValue(1)
        self.changed.emit()
        
    def set_insight(self, on: bool) -> None:
        self.btn_insight.blockSignals(True)
        try:
            self.btn_insight.setChecked(bool(on))
        finally:
            self.btn_insight.blockSignals(False)

    def _fill_combo(
        self,
        cmb: QComboBox,
        items: List[DatasetChoice],
    ) -> None:
        cmb.blockSignals(True)
        try:
            cmb.clear()
            for it in items:
                cmb.addItem(it.text, it.data)
        finally:
            cmb.blockSignals(False)

