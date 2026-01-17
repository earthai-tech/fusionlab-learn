# geoprior/ui/map/view_panel.py
# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>

"""
geoprior.ui.map.view_panel

Auto-hide view/options panel (C).

Goals
-----
- Modern controls (compact + scrollable).
- Store-backed (single source of truth).
- Safe defaults + "Reset".
- Emits a lightweight signal for "apply".
"""

from __future__ import annotations

from typing import Dict, Optional

from PyQt5.QtCore import Qt, QSignalBlocker, pyqtSignal
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ...config.store import GeoConfigStore
from .data_panel import AutoHidePanel


_VIEW_DEFAULTS: Dict[str, object] = {
    "map.view.basemap": "osm",
    "map.view.basemap_style": "light",
    "map.view.tiles_opacity": 1.0,
    "map.view.colormap": "viridis",
    "map.view.cmap_invert": False,
    "map.view.autoscale": True,
    "map.view.vmin": 0.0,
    "map.view.vmax": 1.0,
    "map.view.clip_mode": "none",
    "map.view.marker_size": 6,
    "map.view.marker_opacity": 0.85,
    "map.view.show_colorbar": True,
    "map.view.show_legend": False,
    "map.view.legend_pos": "br",
    # Hotspots (attention layer)
    "map.view.hotspots.enabled": False,
    "map.view.hotspots.mode": "auto",          # auto|manual|merge
    "map.view.hotspots.method": "grid",        # grid|quantile|cluster
    "map.view.hotspots.metric": "high",        # value|abs|high|low
    "map.view.hotspots.time_agg": "current",   # current|mean|max|trend
    "map.view.hotspots.thr_mode": "quantile",  # quantile|absolute
    "map.view.hotspots.quantile": 0.98,
    "map.view.hotspots.abs_thr": 0.0,
    "map.view.hotspots.cell_km": 1.0,
    "map.view.hotspots.min_pts": 20,
    "map.view.hotspots.max_n": 8,
    "map.view.hotspots.min_sep_km": 2.0,
    
    "map.view.hotspots.style": "pulse",        # pulse|glow
    "map.view.hotspots.pulse": True,
    "map.view.hotspots.pulse_speed": 1.0,
    "map.view.hotspots.ring_km": 0.8,
    "map.view.hotspots.labels": True,
    
    # Interpretation (placeholder but store-backed)
    "map.view.interp.enabled": False,
    "map.view.interp.scheme": "subsidence",    # subsidence|risk
    "map.view.interp.callouts": True,

}


class AutoHideViewPanel(AutoHidePanel):
    """
    Right view panel (C).

    Store keys
    ----------
    map.view.*
    """

    changed = pyqtSignal(object)

    def __init__(
        self,
        *,
        store: GeoConfigStore,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(
            title="View",
            side="right",
            expanded_w=320,
            parent=parent,
        )
        self.store = store

        self._build_body()
        self._apply_defaults()
        self._sync_from_store()
        self._connect()

    # -------------------------------------------------
    # UI
    # -------------------------------------------------
    def _build_body(self) -> None:
        root = QVBoxLayout(self.body)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(10)

        bar = QWidget(self.body)
        bl = QHBoxLayout(bar)
        bl.setContentsMargins(0, 0, 0, 0)
        bl.setSpacing(8)

        self.lb_hint = QLabel("Map style & rendering", bar)
        self.lb_hint.setStyleSheet("font-weight:600;")

        self.btn_reset = QPushButton("Reset", bar)
        self.btn_reset.setToolTip("Restore defaults")

        self.btn_apply = QToolButton(bar)
        self.btn_apply.setText("✓")
        self.btn_apply.setToolTip("Apply to map")

        bl.addWidget(self.lb_hint, 1)
        bl.addWidget(self.btn_reset, 0)
        bl.addWidget(self.btn_apply, 0)

        root.addWidget(bar, 0)

        self.scroll = QScrollArea(self.body)
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.NoFrame)

        host = QWidget(self.scroll)
        self.scroll.setWidget(host)

        lay = QVBoxLayout(host)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(10)

        lay.addWidget(self._group_basemap(host), 0)
        lay.addWidget(self._group_colors(host), 0)
        lay.addWidget(self._group_markers(host), 0)
        lay.addWidget(self._group_legend(host), 0)
        
        lay.addWidget(self._group_hotspots(host), 0)
        lay.addWidget(self._group_interpretation(host), 0)
        
        lay.addStretch(1)

        root.addWidget(self.scroll, 1)

    def _group_basemap(self, parent: QWidget) -> QGroupBox:
        box = QGroupBox("Basemap", parent)
        form = QFormLayout(box)
        form.setContentsMargins(10, 10, 10, 10)
        form.setHorizontalSpacing(10)
        form.setVerticalSpacing(8)

        self.cmb_base = QComboBox(box)
        self.cmb_base.addItems([
            "osm",
            "terrain",
            "satellite",
        ])
        self.cmb_base.setToolTip(
            "Base tiles provider (engine dependent)."
        )

        self.cmb_style = QComboBox(box)
        self.cmb_style.addItems([
            "light",
            "dark",
            "gray",
        ])
        self.cmb_style.setToolTip("Tiles style preset")

        self.sl_tiles_op = QSlider(Qt.Horizontal, box)
        self.sl_tiles_op.setRange(0, 100)
        self.sl_tiles_op.setSingleStep(1)
        self.sl_tiles_op.setToolTip("Tiles opacity")

        self.lb_tiles_op = QLabel("100%", box)
        self.lb_tiles_op.setMinimumWidth(44)
        self.lb_tiles_op.setAlignment(
            Qt.AlignRight | Qt.AlignVCenter
        )

        row = QWidget(box)
        rl = QHBoxLayout(row)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.setSpacing(8)
        rl.addWidget(self.sl_tiles_op, 1)
        rl.addWidget(self.lb_tiles_op, 0)

        form.addRow("Provider", self.cmb_base)
        form.addRow("Style", self.cmb_style)
        form.addRow("Opacity", row)

        return box

    def _group_colors(self, parent: QWidget) -> QGroupBox:
        box = QGroupBox("Color mapping", parent)
        form = QFormLayout(box)
        form.setContentsMargins(10, 10, 10, 10)
        form.setHorizontalSpacing(10)
        form.setVerticalSpacing(8)

        self.cmb_cmap = QComboBox(box)
        self.cmb_cmap.addItems([
            "viridis",
            "plasma",
            "inferno",
            "magma",
            "cividis",
            "turbo",
        ])
        self.cmb_cmap.setToolTip("Colormap name")

        self.chk_inv = QCheckBox("Invert", box)
        self.chk_inv.setToolTip("Reverse colormap")

        top = QWidget(box)
        tl = QHBoxLayout(top)
        tl.setContentsMargins(0, 0, 0, 0)
        tl.setSpacing(10)
        tl.addWidget(self.cmb_cmap, 1)
        tl.addWidget(self.chk_inv, 0)

        self.cmb_clip = QComboBox(box)
        self.cmb_clip.addItems([
            "none",
            "p02-p98",
            "p05-p95",
            "p10-p90",
        ])
        self.cmb_clip.setToolTip(
            "Clip values before scaling (robust)."
        )

        self.chk_auto = QCheckBox("Autoscale", box)
        self.chk_auto.setToolTip("Compute vmin/vmax from data")

        self.sp_vmin = QDoubleSpinBox(box)
        self.sp_vmin.setDecimals(6)
        self.sp_vmin.setRange(-1e12, 1e12)
        self.sp_vmin.setSingleStep(0.1)

        self.sp_vmax = QDoubleSpinBox(box)
        self.sp_vmax.setDecimals(6)
        self.sp_vmax.setRange(-1e12, 1e12)
        self.sp_vmax.setSingleStep(0.1)

        form.addRow("Colormap", top)
        form.addRow("Clip", self.cmb_clip)
        form.addRow("", self.chk_auto)
        form.addRow("Vmin", self.sp_vmin)
        form.addRow("Vmax", self.sp_vmax)

        return box

    def _group_markers(self, parent: QWidget) -> QGroupBox:
        box = QGroupBox("Markers", parent)
        form = QFormLayout(box)
        form.setContentsMargins(10, 10, 10, 10)
        form.setHorizontalSpacing(10)
        form.setVerticalSpacing(8)

        self.sp_size = QSpinBox(box)
        self.sp_size.setRange(1, 40)
        self.sp_size.setSingleStep(1)
        self.sp_size.setToolTip("Marker size (pixels)")

        self.sl_op = QSlider(Qt.Horizontal, box)
        self.sl_op.setRange(0, 100)
        self.sl_op.setSingleStep(1)
        self.sl_op.setToolTip("Marker opacity")

        self.lb_op = QLabel("85%", box)
        self.lb_op.setMinimumWidth(44)
        self.lb_op.setAlignment(
            Qt.AlignRight | Qt.AlignVCenter
        )

        row = QWidget(box)
        rl = QHBoxLayout(row)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.setSpacing(8)
        rl.addWidget(self.sl_op, 1)
        rl.addWidget(self.lb_op, 0)

        form.addRow("Size", self.sp_size)
        form.addRow("Opacity", row)

        return box

    def _group_legend(self, parent: QWidget) -> QGroupBox:
        box = QGroupBox("Legend", parent)
        form = QFormLayout(box)
        form.setContentsMargins(10, 10, 10, 10)
        form.setHorizontalSpacing(10)
        form.setVerticalSpacing(8)

        self.chk_cbar = QCheckBox("Show colorbar", box)
        self.chk_leg = QCheckBox("Show legend", box)

        self.cmb_legpos = QComboBox(box)
        self.cmb_legpos.addItems([
            "br",  # bottom-right
            "bl",
            "tr",
            "tl",
        ])
        self.cmb_legpos.setToolTip("Legend position")

        form.addRow("", self.chk_cbar)
        form.addRow("", self.chk_leg)
        form.addRow("Position", self.cmb_legpos)

        return box
    
    def _group_hotspots(self, parent: QWidget) -> QGroupBox:
        box = QGroupBox("Hotspots", parent)
        form = QFormLayout(box)
        form.setContentsMargins(10, 10, 10, 10)
        form.setHorizontalSpacing(10)
        form.setVerticalSpacing(8)
    
        self.chk_hot = QCheckBox("Enable hotspots", box)
    
        self.cmb_hot_mode = QComboBox(box)
        self.cmb_hot_mode.addItems([
            "auto",
            "manual",
            "merge",
        ])
    
        self.cmb_hot_method = QComboBox(box)
        self.cmb_hot_method.addItems([
            "grid",
            "quantile",
            "cluster",
        ])
    
        self.cmb_hot_metric = QComboBox(box)
        self.cmb_hot_metric.addItems([
            "value",
            "abs",
            "high",
            "low",
        ])
    
        self.cmb_hot_time = QComboBox(box)
        self.cmb_hot_time.addItems([
            "current",
            "mean",
            "max",
            "trend",
        ])
        self.cmb_hot_time.setToolTip(
            "Time aggregation (v1 uses current selection)."
        )
    
        self.cmb_thr = QComboBox(box)
        self.cmb_thr.addItems(["quantile", "absolute"])
    
        # Quantile slider (0.90..0.995)
        self.sl_q = QSlider(Qt.Horizontal, box)
        self.sl_q.setRange(900, 995)
        self.sl_q.setSingleStep(1)
        self.lb_q = QLabel("0.980", box)
        self.lb_q.setMinimumWidth(54)
        self.lb_q.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
    
        qrow = QWidget(box)
        ql = QHBoxLayout(qrow)
        ql.setContentsMargins(0, 0, 0, 0)
        ql.setSpacing(8)
        ql.addWidget(self.sl_q, 1)
        ql.addWidget(self.lb_q, 0)
    
        self.sp_abs = QDoubleSpinBox(box)
        self.sp_abs.setDecimals(6)
        self.sp_abs.setRange(-1e12, 1e12)
        self.sp_abs.setSingleStep(0.1)
    
        self.sp_cell = QDoubleSpinBox(box)
        self.sp_cell.setDecimals(3)
        self.sp_cell.setRange(0.01, 1e6)
        self.sp_cell.setSingleStep(0.1)
    
        self.sp_minpts = QSpinBox(box)
        self.sp_minpts.setRange(1, 1_000_000)
    
        self.sp_maxn = QSpinBox(box)
        self.sp_maxn.setRange(1, 100)
    
        self.sp_sep = QDoubleSpinBox(box)
        self.sp_sep.setDecimals(2)
        self.sp_sep.setRange(0.0, 1e6)
        self.sp_sep.setSingleStep(0.25)
    
        # Visual emphasis
        self.cmb_hot_style = QComboBox(box)
        self.cmb_hot_style.addItems(["pulse", "glow"])
    
        self.chk_pulse = QCheckBox("Pulse", box)
    
        self.sl_speed = QSlider(Qt.Horizontal, box)
        self.sl_speed.setRange(20, 300)  # -> 0.2..3.0
        self.sl_speed.setSingleStep(5)
        self.lb_speed = QLabel("1.0×", box)
        self.lb_speed.setMinimumWidth(54)
        self.lb_speed.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
    
        srow = QWidget(box)
        sl = QHBoxLayout(srow)
        sl.setContentsMargins(0, 0, 0, 0)
        sl.setSpacing(8)
        sl.addWidget(self.sl_speed, 1)
        sl.addWidget(self.lb_speed, 0)
    
        self.sp_ring = QDoubleSpinBox(box)
        self.sp_ring.setDecimals(2)
        self.sp_ring.setRange(0.01, 1000.0)
        self.sp_ring.setSingleStep(0.1)
    
        self.chk_labels = QCheckBox("Labels", box)
    
        form.addRow("", self.chk_hot)
        form.addRow("Mode", self.cmb_hot_mode)
        form.addRow("Method", self.cmb_hot_method)
        form.addRow("Metric", self.cmb_hot_metric)
        form.addRow("Time", self.cmb_hot_time)
    
        form.addRow("Threshold", self.cmb_thr)
        form.addRow("Quantile", qrow)
        form.addRow("Abs thr", self.sp_abs)
    
        form.addRow("Cell (km)", self.sp_cell)
        form.addRow("Min pts", self.sp_minpts)
        form.addRow("Max N", self.sp_maxn)
        form.addRow("Min sep (km)", self.sp_sep)
    
        form.addRow("Style", self.cmb_hot_style)
        form.addRow("", self.chk_pulse)
        form.addRow("Speed", srow)
        form.addRow("Ring (km)", self.sp_ring)
        form.addRow("", self.chk_labels)
    
        return box
    
    def _group_interpretation(self, parent: QWidget) -> QGroupBox:
        box = QGroupBox("Interpretation", parent)
        form = QFormLayout(box)
        form.setContentsMargins(10, 10, 10, 10)
        form.setHorizontalSpacing(10)
        form.setVerticalSpacing(8)
    
        self.chk_interp = QCheckBox("Enable interpretation", box)
    
        self.cmb_interp = QComboBox(box)
        self.cmb_interp.addItems([
            "subsidence",
            "risk",
        ])
    
        self.chk_callouts = QCheckBox(
            "Show callouts on hotspots",
            box,
        )
    
        form.addRow("", self.chk_interp)
        form.addRow("Scheme", self.cmb_interp)
        form.addRow("", self.chk_callouts)
    
        return box

    # -------------------------------------------------
    # Store sync
    # -------------------------------------------------
    def _apply_defaults(self) -> None:
        with self.store.batch():
            for k, v in _VIEW_DEFAULTS.items():
                cur = self.store.get(k, None)
                if cur is None or cur == "":
                    self.store.set(k, v)

    def _sync_from_store(self) -> None:
        blk = QSignalBlocker(self)

        _ = blk  # keep reference

        self._set_combo(
            self.cmb_base,
            str(self.store.get(
                "map.view.basemap",
                "osm",
            )),
        )
        self._set_combo(
            self.cmb_style,
            str(self.store.get(
                "map.view.basemap_style",
                "light",
            )),
        )

        top = float(self.store.get(
            "map.view.tiles_opacity",
            1.0,
        ) or 1.0)
        self._set_slider_pct(self.sl_tiles_op, top)
        self.lb_tiles_op.setText(self._pct_text(top))

        self._set_combo(
            self.cmb_cmap,
            str(self.store.get(
                "map.view.colormap",
                "viridis",
            )),
        )
        self.chk_inv.setChecked(bool(self.store.get(
            "map.view.cmap_invert",
            False,
        )))

        self._set_combo(
            self.cmb_clip,
            str(self.store.get(
                "map.view.clip_mode",
                "none",
            )),
        )
        self.chk_auto.setChecked(bool(self.store.get(
            "map.view.autoscale",
            True,
        )))

        self.sp_vmin.setValue(float(self.store.get(
            "map.view.vmin",
            0.0,
        ) or 0.0))
        self.sp_vmax.setValue(float(self.store.get(
            "map.view.vmax",
            1.0,
        ) or 1.0))

        size = int(self.store.get(
            "map.view.marker_size",
            6,
        ) or 6)
        self.sp_size.setValue(size)

        op = float(self.store.get(
            "map.view.marker_opacity",
            0.85,
        ) or 0.85)
        self._set_slider_pct(self.sl_op, op)
        self.lb_op.setText(self._pct_text(op))

        self.chk_cbar.setChecked(bool(self.store.get(
            "map.view.show_colorbar",
            True,
        )))
        self.chk_leg.setChecked(bool(self.store.get(
            "map.view.show_legend",
            False,
        )))
        self._set_combo(
            self.cmb_legpos,
            str(self.store.get(
                "map.view.legend_pos",
                "br",
            )),
        )
        
        # --- hotspots
        self.chk_hot.setChecked(bool(self.store.get(
            "map.view.hotspots.enabled", False
        )))
        self._set_combo(self.cmb_hot_mode, str(self.store.get(
            "map.view.hotspots.mode", "auto"
        )))
        self._set_combo(self.cmb_hot_method, str(self.store.get(
            "map.view.hotspots.method", "grid"
        )))
        self._set_combo(self.cmb_hot_metric, str(self.store.get(
            "map.view.hotspots.metric", "high"
        )))
        self._set_combo(self.cmb_hot_time, str(self.store.get(
            "map.view.hotspots.time_agg", "current"
        )))
        self._set_combo(self.cmb_thr, str(self.store.get(
            "map.view.hotspots.thr_mode", "quantile"
        )))
        
        q = float(self.store.get("map.view.hotspots.quantile", 0.98) or 0.98)
        q = max(0.0, min(1.0, q))
        self.sl_q.setValue(int(round(q * 1000.0)))
        self.lb_q.setText(f"{q:.3f}")
        
        self.sp_abs.setValue(float(self.store.get(
            "map.view.hotspots.abs_thr", 0.0
        ) or 0.0))
        
        self.sp_cell.setValue(float(self.store.get(
            "map.view.hotspots.cell_km", 1.0
        ) or 1.0))
        self.sp_minpts.setValue(int(self.store.get(
            "map.view.hotspots.min_pts", 20
        ) or 20))
        self.sp_maxn.setValue(int(self.store.get(
            "map.view.hotspots.max_n", 8
        ) or 8))
        self.sp_sep.setValue(float(self.store.get(
            "map.view.hotspots.min_sep_km", 2.0
        ) or 2.0))
        
        self._set_combo(self.cmb_hot_style, str(self.store.get(
            "map.view.hotspots.style", "pulse"
        )))
        self.chk_pulse.setChecked(bool(self.store.get(
            "map.view.hotspots.pulse", True
        )))
        
        sp = float(self.store.get("map.view.hotspots.pulse_speed", 1.0) or 1.0)
        sp = max(0.2, min(3.0, sp))
        self.sl_speed.setValue(int(round(sp * 100.0)))
        self.lb_speed.setText(f"{sp:.1f}×")
        
        rk = float(self.store.get("map.view.hotspots.ring_km", 0.8) or 0.8)
        self.sp_ring.setValue(rk)
        
        self.chk_labels.setChecked(bool(self.store.get(
            "map.view.hotspots.labels", True
        )))
        
        # --- interpretation
        self.chk_interp.setChecked(bool(self.store.get(
            "map.view.interp.enabled", False
        )))
        self._set_combo(self.cmb_interp, str(self.store.get(
            "map.view.interp.scheme", "subsidence"
        )))
        self.chk_callouts.setChecked(bool(self.store.get(
            "map.view.interp.callouts", True
        )))
        
        self._update_hotspot_ui_enabled()

        self._update_vrange_enabled()

    def _update_hotspot_ui_enabled(self) -> None:
        on = bool(self.chk_hot.isChecked())
        mode = str(self.cmb_hot_mode.currentText() or "auto").lower()
        thr = str(self.cmb_thr.currentText() or "quantile").lower()
    
        # enable/disable whole set (except master checkbox)
        for w in (
            self.cmb_hot_mode,
            self.cmb_hot_method,
            self.cmb_hot_metric,
            self.cmb_hot_time,
            self.cmb_thr,
            self.sl_q,
            self.sp_abs,
            self.sp_cell,
            self.sp_minpts,
            self.sp_maxn,
            self.sp_sep,
            self.cmb_hot_style,
            self.chk_pulse,
            self.sl_speed,
            self.sp_ring,
            self.chk_labels,
        ):
            w.setEnabled(on)
    
        # Manual mode: auto params are disabled
        auto_on = on and (mode != "manual")
        for w in (
            self.cmb_hot_method,
            self.cmb_hot_metric,
            self.cmb_hot_time,
            self.cmb_thr,
            self.sl_q,
            self.sp_abs,
            self.sp_cell,
            self.sp_minpts,
            self.sp_maxn,
            self.sp_sep,
        ):
            w.setEnabled(auto_on)
    
        # Threshold: quantile vs absolute
        self.sl_q.setEnabled(auto_on and thr == "quantile")
        self.sp_abs.setEnabled(auto_on and thr == "absolute")
    
        # Pulse speed only matters if pulse + style pulse
        style = str(self.cmb_hot_style.currentText() or "pulse").lower()
        pulse_on = bool(self.chk_pulse.isChecked()) and style == "pulse"
        self.sl_speed.setEnabled(on and pulse_on)

    def _connect(self) -> None:
        self.btn_reset.clicked.connect(self._on_reset)
        self.btn_apply.clicked.connect(self._emit_changed)

        self.cmb_base.currentTextChanged.connect(
            lambda v: self.store.set(
                "map.view.basemap", str(v)
            )
        )
        self.cmb_style.currentTextChanged.connect(
            lambda v: self.store.set(
                "map.view.basemap_style", str(v)
            )
        )
        self.sl_tiles_op.valueChanged.connect(
            self._on_tiles_op
        )

        self.cmb_cmap.currentTextChanged.connect(
            lambda v: self.store.set(
                "map.view.colormap", str(v)
            )
        )
        self.chk_inv.toggled.connect(
            lambda b: self.store.set(
                "map.view.cmap_invert", bool(b)
            )
        )
        self.cmb_clip.currentTextChanged.connect(
            lambda v: self.store.set(
                "map.view.clip_mode", str(v)
            )
        )

        self.chk_auto.toggled.connect(self._on_auto)
        self.sp_vmin.valueChanged.connect(self._on_vmin)
        self.sp_vmax.valueChanged.connect(self._on_vmax)

        self.sp_size.valueChanged.connect(
            lambda v: self.store.set(
                "map.view.marker_size", int(v)
            )
        )
        self.sl_op.valueChanged.connect(self._on_op)

        self.chk_cbar.toggled.connect(
            lambda b: self.store.set(
                "map.view.show_colorbar", bool(b)
            )
        )
        self.chk_leg.toggled.connect(
            lambda b: self.store.set(
                "map.view.show_legend", bool(b)
            )
        )
        self.cmb_legpos.currentTextChanged.connect(
            lambda v: self.store.set(
                "map.view.legend_pos", str(v)
            )
        )

        self.store.config_changed.connect(
            self._on_store_changed,
        )
        
        # --- hotspots
        self.chk_hot.toggled.connect(
            lambda b: (
                self.store.set("map.view.hotspots.enabled", bool(b)),
                self._update_hotspot_ui_enabled(),
            )
        )
        
        self.cmb_hot_mode.currentTextChanged.connect(
            lambda v: (
                self.store.set("map.view.hotspots.mode", str(v)),
                self._update_hotspot_ui_enabled(),
            )
        )
        
        self.cmb_hot_method.currentTextChanged.connect(
            lambda v: self.store.set(
                "map.view.hotspots.method", str(v)
            )
        )
        self.cmb_hot_metric.currentTextChanged.connect(
            lambda v: self.store.set(
                "map.view.hotspots.metric", str(v)
            )
        )
        self.cmb_hot_time.currentTextChanged.connect(
            lambda v: self.store.set(
                "map.view.hotspots.time_agg", str(v)
            )
        )
        self.cmb_thr.currentTextChanged.connect(
            lambda v: (
                self.store.set("map.view.hotspots.thr_mode", str(v)),
                self._update_hotspot_ui_enabled(),
            )
        )
        
        def _on_q(v: int) -> None:
            q = float(v) / 1000.0
            self.lb_q.setText(f"{q:.3f}")
            self.store.set("map.view.hotspots.quantile", q)
        
        self.sl_q.valueChanged.connect(_on_q)
        
        self.sp_abs.valueChanged.connect(
            lambda v: self.store.set(
                "map.view.hotspots.abs_thr", float(v)
            )
        )
        
        self.sp_cell.valueChanged.connect(
            lambda v: self.store.set(
                "map.view.hotspots.cell_km", float(v)
            )
        )
        self.sp_minpts.valueChanged.connect(
            lambda v: self.store.set(
                "map.view.hotspots.min_pts", int(v)
            )
        )
        self.sp_maxn.valueChanged.connect(
            lambda v: self.store.set(
                "map.view.hotspots.max_n", int(v)
            )
        )
        self.sp_sep.valueChanged.connect(
            lambda v: self.store.set(
                "map.view.hotspots.min_sep_km", float(v)
            )
        )
        
        self.cmb_hot_style.currentTextChanged.connect(
            lambda v: (
                self.store.set("map.view.hotspots.style", str(v)),
                self._update_hotspot_ui_enabled(),
            )
        )
        self.chk_pulse.toggled.connect(
            lambda b: (
                self.store.set("map.view.hotspots.pulse", bool(b)),
                self._update_hotspot_ui_enabled(),
            )
        )
        
        def _on_speed(v: int) -> None:
            sp = float(v) / 100.0
            sp = max(0.2, min(3.0, sp))
            self.lb_speed.setText(f"{sp:.1f}×")
            self.store.set("map.view.hotspots.pulse_speed", sp)
        
        self.sl_speed.valueChanged.connect(_on_speed)
        
        self.sp_ring.valueChanged.connect(
            lambda v: self.store.set(
                "map.view.hotspots.ring_km", float(v)
            )
        )
        self.chk_labels.toggled.connect(
            lambda b: self.store.set(
                "map.view.hotspots.labels", bool(b)
            )
        )
        
        # --- interpretation
        self.chk_interp.toggled.connect(
            lambda b: self.store.set(
                "map.view.interp.enabled", bool(b)
            )
        )
        self.cmb_interp.currentTextChanged.connect(
            lambda v: self.store.set(
                "map.view.interp.scheme", str(v)
            )
        )
        self.chk_callouts.toggled.connect(
            lambda b: self.store.set(
                "map.view.interp.callouts", bool(b)
            )
        )

    # -------------------------------------------------
    # Handlers
    # -------------------------------------------------
    def _on_store_changed(self, keys) -> None:
        keys = set(keys or [])
        if not keys:
            return
        if any(k.startswith("map.view.") for k in keys):
            self._sync_from_store()

    def _on_reset(self) -> None:
        with self.store.batch():
            for k, v in _VIEW_DEFAULTS.items():
                self.store.set(k, v)
        self._sync_from_store()
        self._emit_changed()

    def _emit_changed(self) -> None:
        self.changed.emit(self.snapshot())

    def _on_tiles_op(self, v: int) -> None:
        f = float(v) / 100.0
        self.lb_tiles_op.setText(self._pct_text(f))
        self.store.set("map.view.tiles_opacity", f)

    def _on_op(self, v: int) -> None:
        f = float(v) / 100.0
        self.lb_op.setText(self._pct_text(f))
        self.store.set("map.view.marker_opacity", f)

    def _on_auto(self, enabled: bool) -> None:
        self.store.set("map.view.autoscale", bool(enabled))
        self._update_vrange_enabled()

    def _on_vmin(self, v: float) -> None:
        self.store.set("map.view.vmin", float(v))

    def _on_vmax(self, v: float) -> None:
        self.store.set("map.view.vmax", float(v))

    def _update_vrange_enabled(self) -> None:
        auto = bool(self.chk_auto.isChecked())
        self.sp_vmin.setEnabled(not auto)
        self.sp_vmax.setEnabled(not auto)

    # -------------------------------------------------
    # Public helpers
    # -------------------------------------------------
    def snapshot(self) -> Dict[str, object]:
        keys = list(_VIEW_DEFAULTS.keys())
        return {k: self.store.get(k, None) for k in keys}

    # -------------------------------------------------
    # Small UI helpers
    # -------------------------------------------------
    def _set_combo(self, cmb: QComboBox, v: str) -> None:
        v = str(v or "").strip()
        if not v:
            return
        idx = cmb.findText(v, Qt.MatchFixedString)
        if idx >= 0:
            cmb.setCurrentIndex(idx)

    def _set_slider_pct(self, sl: QSlider, f: float) -> None:
        f = float(f)
        v = int(max(0.0, min(1.0, f)) * 100.0)
        sl.setValue(v)

    def _pct_text(self, f: float) -> str:
        v = int(max(0.0, min(1.0, float(f))) * 100.0)
        return f"{v}%"
