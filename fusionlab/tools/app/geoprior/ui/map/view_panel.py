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

from PyQt5.QtGui import QIcon
from PyQt5.QtCore import Qt, QSignalBlocker, pyqtSignal
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
    QStyle
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
    
    # Interpretation (policy-ready, store-backed)
    "map.view.interp.enabled": False,
    "map.view.interp.scheme": "subsidence",
    "map.view.interp.callouts": True,
    "map.view.interp.callout_level": "standard",
    "map.view.interp.callout_actions": True,
    "map.view.interp.action_pack": "balanced",
    "map.view.interp.action_intensity": "balanced",
    "map.view.interp.summary": "",

}


class AutoHideViewPanel(AutoHidePanel):
    """
    Right view panel (C).

    Store keys
    ----------
    map.view.*
    """

    changed = pyqtSignal(object)
    export_requested = pyqtSignal(str)

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

    def _std_icon(self, sp: QStyle.StandardPixmap) -> QIcon:
        return self.style().standardIcon(sp)

    def _make_card(
        self,
        parent: QWidget,
        *,
        title: str,
        sp: QStyle.StandardPixmap,
    ) -> tuple[QFrame, QWidget, QLabel]:
        card = QFrame(parent)
        card.setObjectName("mapPanelCard")

        root = QVBoxLayout(card)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        head = QWidget(card)
        hl = QHBoxLayout(head)
        hl.setContentsMargins(0, 0, 0, 0)
        hl.setSpacing(8)

        ico = QLabel(head)
        ico.setPixmap(self._std_icon(sp).pixmap(16, 16))

        lb = QLabel(title, head)
        lb.setObjectName("mapSectionTitle")

        chip = QLabel("", head)
        chip.setObjectName("mapCountChip")

        hl.addWidget(ico, 0)
        hl.addWidget(lb, 1)
        hl.addWidget(chip, 0)

        body = QWidget(card)

        root.addWidget(head, 0)
        root.addWidget(body, 0)

        return card, body, chip


    def _build_body(self) -> None:
        root = QVBoxLayout(self.body)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(10)

        bar = QFrame(self.body)
        bar.setObjectName("mapPanelCard")
        bar.setProperty("role", "toolbar")

        bl = QHBoxLayout(bar)
        bl.setContentsMargins(10, 8, 10, 8)
        bl.setSpacing(8)

        self.lb_hint = QLabel("View & rendering", bar)
        self.lb_hint.setObjectName("mapSectionTitle")

        self.btn_reset = QToolButton(bar)
        self.btn_reset.setObjectName("miniAction")
        self.btn_reset.setToolTip("Restore defaults")
        self.btn_reset.setAutoRaise(True)
        self.btn_reset.setIcon(
            self._std_icon(QStyle.SP_BrowserReload)
        )
        self.btn_reset.setText("Reset")
        self.btn_reset.setToolButtonStyle(
            Qt.ToolButtonTextBesideIcon
        )

        self.btn_apply = QToolButton(bar)
        self.btn_apply.setObjectName("miniAction")
        self.btn_apply.setToolTip("Apply to map")
        self.btn_apply.setAutoRaise(True)
        self.btn_apply.setIcon(
            self._std_icon(QStyle.SP_DialogApplyButton)
        )
        self.btn_apply.setText("")
        self.btn_apply.setToolButtonStyle(
            Qt.ToolButtonIconOnly
        )

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

    def _group_basemap(self, parent: QWidget) -> QFrame:
        box, body, chip = self._make_card(
            parent,
            title="Basemap",
            sp=QStyle.SP_DirIcon,
        )
        self._chip_base = chip
        
        form = QFormLayout(body)
        form.setContentsMargins(0, 0, 0, 0)
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

    def _group_colors(self, parent: QWidget) -> QFrame:
        box, body, chip = self._make_card(
            parent,
            title="Color mapping",
            sp=QStyle.SP_DriveDVDIcon,
        )
        self._chip_base = chip
        
        form = QFormLayout(body)
        form.setContentsMargins(0, 0, 0, 0)

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

    def _group_markers(self, parent: QWidget) -> QFrame:
        box, body, chip = self._make_card(
            parent,
            title="Markers",
            sp=QStyle.SP_FileIcon,
        )
        self._chip_base = chip
        
        form = QFormLayout(body)
        form.setContentsMargins(0, 0, 0, 0)

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

    def _group_legend(self, parent: QWidget) -> QFrame:
        box, body, chip = self._make_card(
            parent,
            title="Legend",
            sp=QStyle.SP_FileDialogDetailedView,
        )
        self._chip_base = chip
        
        form = QFormLayout(body)
        form.setContentsMargins(0, 0, 0, 0)

        
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
    
    def _group_hotspots(self, parent: QWidget) -> QFrame:
        box, body, chip = self._make_card(
            parent,
            title="Hotspots",
            sp=QStyle.SP_ArrowUp,
        )
        self._chip_base = chip
        form = QFormLayout(body)
        form.setContentsMargins(0, 0, 0, 0)
        
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
    
    def _group_interpretation(
        self,
        parent: QWidget,
    ) -> QFrame:
        box, body, chip = self._make_card(
            parent,
            title="Interpretation",
            sp=QStyle.SP_MessageBoxInformation,
        )
        self._chip_base = chip
        form = QFormLayout(body)
        form.setContentsMargins(0, 0, 0, 0)
        
        form.setContentsMargins(10, 10, 10, 10)
        form.setHorizontalSpacing(10)
        form.setVerticalSpacing(8)

        self.chk_interp = QCheckBox(
            "Enable interpretation",
            box,
        )

        self.cmb_interp = QComboBox(box)
        self.cmb_interp.addItems([
            "subsidence",
            "risk",
        ])

        self.cmb_tone = QComboBox(box)
        self.cmb_tone.addItems([
            "municipal",
            "technical",
            "public",
        ])

        self.chk_callouts = QCheckBox(
            "Show callouts on hotspots",
            box,
        )

        self.cmb_call_lvl = QComboBox(box)
        self.cmb_call_lvl.addItems([
            "compact",
            "standard",
            "detailed",
        ])

        self.chk_call_act = QCheckBox(
            "Include actions in callout",
            box,
        )

        self.cmb_pack = QComboBox(box)
        self.cmb_pack.addItems([
            "balanced",
            "groundwater",
            "infrastructure",
            "planning",
            "monitoring",
        ])

        self.cmb_int = QComboBox(box)
        self.cmb_int.addItems([
            "conservative",
            "balanced",
            "aggressive",
        ])

        self.lb_interp_sum = QLabel("", box)
        self.lb_interp_sum.setWordWrap(True)
        self.lb_interp_sum.setStyleSheet(
            "color: rgba(140,140,140,1);"
        )

        exp = QWidget(box)
        el = QHBoxLayout(exp)
        el.setContentsMargins(0, 0, 0, 0)
        el.setSpacing(6)

        self.btn_exp_csv = QPushButton("CSV", box)
        self.btn_exp_geo = QPushButton("GeoJSON", box)
        self.btn_exp_brief = QPushButton("Brief", box)

        self.btn_exp_csv.setToolTip(
            "Export hotspots summary (CSV)"
        )
        self.btn_exp_geo.setToolTip(
            "Export hotspots as GeoJSON"
        )
        self.btn_exp_brief.setToolTip(
            "Export a municipal brief (Markdown)"
        )

        el.addWidget(self.btn_exp_csv, 0)
        el.addWidget(self.btn_exp_geo, 0)
        el.addWidget(self.btn_exp_brief, 0)
        el.addStretch(1)

        form.addRow("", self.chk_interp)
        form.addRow("Scheme", self.cmb_interp)
        form.addRow("Tone", self.cmb_tone)
        form.addRow("", self.chk_callouts)
        form.addRow("Callout", self.cmb_call_lvl)
        form.addRow("", self.chk_call_act)
        form.addRow("Actions", self.cmb_pack)
        form.addRow("Intensity", self.cmb_int)
        form.addRow("Summary", self.lb_interp_sum)
        form.addRow("Export", exp)
        
        return box

    def _update_chips(self) -> None:
        if hasattr(self, "_chip_base"):
            b = str(self.cmb_base.currentText() or "osm")
            s = str(self.cmb_style.currentText() or "light")
            self._chip_base.setText(f"{b}/{s}")
    
        if hasattr(self, "_chip_colors"):
            cmap = str(self.cmb_cmap.currentText() or "viridis")
            inv = "inv" if self.chk_inv.isChecked() else ""
            clip = str(self.cmb_clip.currentText() or "none")
            txt = f"{cmap} {inv}".strip()
            if clip and clip != "none":
                txt = f"{txt} | {clip}".strip()
            self._chip_colors.setText(txt)
    
        if hasattr(self, "_chip_markers"):
            sz = int(self.sp_size.value())
            op = int(self.sl_op.value())
            self._chip_markers.setText(f"{sz}px/{op}%")
    
        if hasattr(self, "_chip_legend"):
            on = "ON" if self.chk_cbar.isChecked() else "OFF"
            pos = str(self.cmb_legpos.currentText() or "br")
            self._chip_legend.setText(f"{on}/{pos}")
    
        if hasattr(self, "_chip_hot"):
            on = "ON" if self.chk_hot.isChecked() else "OFF"
            self._chip_hot.setText(on)
    
        if hasattr(self, "_chip_interp"):
            on = "ON" if self.chk_interp.isChecked() else "OFF"
            self._chip_interp.setText(on)

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
            "map.view.interp.enabled",
            False,
        )))

        self._set_combo(
            self.cmb_interp,
            str(self.store.get(
                "map.view.interp.scheme",
                "subsidence",
            )),
        )

        self.chk_callouts.setChecked(bool(self.store.get(
            "map.view.interp.callouts",
            True,
        )))

        self._set_combo(
            self.cmb_call_lvl,
            str(self.store.get(
                "map.view.interp."
                "callout_level",
                "standard",
            )),
        )

        self.chk_call_act.setChecked(bool(self.store.get(
            "map.view.interp."
            "callout_actions",
            True,
        )))

        self._set_combo(
            self.cmb_tone,
            str(self.store.get(
                "map.view.interp.tone",
                "municipal",
            )),
        )

        self._set_combo(
            self.cmb_pack,
            str(self.store.get(
                "map.view.interp."
                "action_pack",
                "balanced",
            )),
        )

        self._set_combo(
            self.cmb_int,
            str(self.store.get(
                "map.view.interp."
                "action_intensity",
                "balanced",
            )),
        )

        self._update_interp_ui_enabled()
        self._update_interp_summary()

        self._update_hotspot_ui_enabled()
        self._update_vrange_enabled()
        self._update_chips()

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
        
        self._update_chips()


    def _update_interp_ui_enabled(self) -> None:
        interp_on = bool(self.chk_interp.isChecked())
        hot_on = bool(self.chk_hot.isChecked())

        for w in (
            self.cmb_interp,
            self.cmb_tone,
            self.chk_callouts,
            self.cmb_call_lvl,
            self.chk_call_act,
            self.cmb_pack,
            self.cmb_int,
            self.lb_interp_sum,
            self.btn_exp_csv,
            self.btn_exp_geo,
            self.btn_exp_brief,
        ):
            w.setEnabled(interp_on)

        # Callouts + exports are meaningful with hotspots
        for w in (
            self.chk_callouts,
            self.cmb_call_lvl,
            self.chk_call_act,
            self.btn_exp_csv,
            self.btn_exp_geo,
            self.btn_exp_brief,
        ):
            w.setEnabled(interp_on and hot_on)
            
        self._update_chips()


    def _update_interp_summary(self) -> None:
        on = bool(self.chk_interp.isChecked())
        hot = bool(self.chk_hot.isChecked())
        sch = str(self.cmb_interp.currentText() or "")
        tone = str(self.cmb_tone.currentText() or "")
        call = bool(self.chk_callouts.isChecked())
        lvl = str(self.cmb_call_lvl.currentText() or "")

        if not on:
            self.lb_interp_sum.setText("Disabled.")
            return

        if not hot:
            self.lb_interp_sum.setText(
                "Enable hotspots to use callouts/exports."
            )
            return

        msg = f"{sch} | {tone} | callouts={call} | {lvl}"
        self.lb_interp_sum.setText(msg)

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
            lambda b: (
                self.store.set(
                    "map.view.interp.enabled",
                    bool(b),
                ),
                self._update_interp_ui_enabled(),
                self._update_interp_summary(),
            )
        )

        self.cmb_interp.currentTextChanged.connect(
            lambda v: (
                self.store.set(
                    "map.view.interp.scheme",
                    str(v),
                ),
                self._update_interp_summary(),
            )
        )

        self.cmb_tone.currentTextChanged.connect(
            lambda v: (
                self.store.set(
                    "map.view.interp.tone",
                    str(v),
                ),
                self._update_interp_summary(),
            )
        )

        self.chk_callouts.toggled.connect(
            lambda b: (
                self.store.set(
                    "map.view.interp.callouts",
                    bool(b),
                ),
                self._update_interp_summary(),
            )
        )

        self.cmb_call_lvl.currentTextChanged.connect(
            lambda v: (
                self.store.set(
                    "map.view.interp."
                    "callout_level",
                    str(v),
                ),
                self._update_interp_summary(),
            )
        )

        self.chk_call_act.toggled.connect(
            lambda b: self.store.set(
                "map.view.interp."
                "callout_actions",
                bool(b),
            )
        )

        self.cmb_pack.currentTextChanged.connect(
            lambda v: self.store.set(
                "map.view.interp."
                "action_pack",
                str(v),
            )
        )

        self.cmb_int.currentTextChanged.connect(
            lambda v: self.store.set(
                "map.view.interp."
                "action_intensity",
                str(v),
            )
        )

        self.btn_exp_csv.clicked.connect(
            lambda: self.export_requested.emit(
                "hotspots_csv"
            )
        )
        self.btn_exp_geo.clicked.connect(
            lambda: self.export_requested.emit(
                "hotspots_geojson"
            )
        )
        self.btn_exp_brief.clicked.connect(
            lambda: self.export_requested.emit(
                "policy_brief"
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
