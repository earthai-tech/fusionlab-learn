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

        self._update_vrange_enabled()

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
