# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>

"""geoprior.ui.setup.cards.coords

Coordinates & CRS.

Modern UX:
- Two-column layout: settings + preview.
- Effective CRS preview + status badge.
- Compact advanced toggles in an expander.
- Store-driven via Binder.
"""

from __future__ import annotations

from typing import Optional

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QSpinBox,
    QToolButton,
    QVBoxLayout,
    QPushButton,
    QWidget,
)

from .base import CardBase
from ..bindings import Binder
from ..schema import default_fields
from ....config.store import GeoConfigStore


class _Expander(QWidget):
    def __init__(
        self,
        title: str,
        *,
        parent: QWidget,
    ) -> None:
        super().__init__(parent)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(6)

        self.btn = QToolButton(self)
        self.btn.setCheckable(True)
        self.btn.setChecked(False)
        self.btn.setText(str(title))
        self.btn.setToolButtonStyle(
            Qt.ToolButtonTextBesideIcon
        )
        self.btn.setArrowType(Qt.RightArrow)

        self.body = QWidget(self)
        self.body.setVisible(False)

        self.body_l = QGridLayout(self.body)
        self.body_l.setContentsMargins(8, 6, 8, 6)
        self.body_l.setHorizontalSpacing(10)
        self.body_l.setVerticalSpacing(6)

        self.btn.toggled.connect(self._toggle)

        root.addWidget(self.btn, 0)
        root.addWidget(self.body, 0)

    def _toggle(self, on: bool) -> None:
        self.body.setVisible(bool(on))
        self.btn.setArrowType(
            Qt.DownArrow if on else Qt.RightArrow
        )


class CoordsCard(CardBase):
    """Coordinates & CRS (store-driven)."""

    def __init__(
        self,
        *,
        store: GeoConfigStore,
        binder: Binder,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(
            section_id="coords",
            title="Coordinates & CRS",
            subtitle=(
                "Define coordinate strategy and EPSG "
                "codes for conversion and modelling."
            ),
            parent=parent,
        )

        self.store = store
        self.binder = binder
        self._fs = default_fields()

        self._build()
        self._wire()
        self.refresh()

    # -----------------------------------------------------------------
    # UI
    # -----------------------------------------------------------------
    def _build(self) -> None:
        body = self.body_layout()

        grid = QWidget(self)
        g = QGridLayout(grid)
        g.setContentsMargins(0, 0, 0, 0)
        g.setHorizontalSpacing(12)
        g.setVerticalSpacing(10)
        g.setColumnStretch(0, 1)
        g.setColumnStretch(1, 1)

        self.grp_core = self._build_core(grid)
        self.grp_preview = self._build_preview(grid)

        g.addWidget(self.grp_core, 0, 0)
        g.addWidget(self.grp_preview, 0, 1)

        body.addWidget(grid, 0)

    def _build_core(self, parent: QWidget) -> QGroupBox:
        box = QGroupBox("CRS settings", parent)
        lay = QGridLayout(box)
        lay.setContentsMargins(10, 10, 10, 10)
        lay.setHorizontalSpacing(10)
        lay.setVerticalSpacing(8)
        lay.setColumnStretch(1, 1)
        lay.setColumnStretch(3, 1)

        r = 0

        # coord_mode
        s = self._fs["coord_mode"]
        self.cmb_mode = QComboBox(box)
        items = s.items if s.items is not None else []

        self.binder.bind_combo(
            "coord_mode",
            self.cmb_mode,
            items=items,
            editable=False,
            none_text=s.none_text,
            use_item_data=True,
        )

        lay.addWidget(self._lab("coord_mode", box), r, 0)
        lay.addWidget(self.cmb_mode, r, 1, 1, 3)
        r += 1

        # EPSG pair
        self.sp_src = QSpinBox(box)
        self.sp_src.setRange(0, 999999)

        self.sp_utm = QSpinBox(box)
        self.sp_utm.setRange(0, 999999)

        self.binder.bind_spin_box(
            "coord_src_epsg",
            self.sp_src,
        )
        self.binder.bind_spin_box("utm_epsg", self.sp_utm)

        lay.addWidget(self._lab("coord_src_epsg", box), r, 0)
        lay.addWidget(self.sp_src, r, 1)
        lay.addWidget(self._lab("utm_epsg", box), r, 2)
        lay.addWidget(self.sp_utm, r, 3)
        r += 1

        # Advanced toggles
        self.exp = _Expander("Advanced", parent=box)

        chk_norm = QCheckBox(self._fs["normalize_coords"].label, box)
        chk_raw = QCheckBox(self._fs["keep_coords_raw"].label, box)
        chk_shift = QCheckBox(self._fs["shift_raw_coords"].label, box)

        for k, w in (
            ("normalize_coords", chk_norm),
            ("keep_coords_raw", chk_raw),
            ("shift_raw_coords", chk_shift),
        ):
            tip = self._fs[k].tooltip
            if tip:
                w.setToolTip(tip)

        self.binder.bind_checkbox("normalize_coords", chk_norm)
        self.binder.bind_checkbox("keep_coords_raw", chk_raw)
        self.binder.bind_checkbox("shift_raw_coords", chk_shift)

        self.exp.body_l.addWidget(chk_norm, 0, 0, 1, 2)
        self.exp.body_l.addWidget(chk_raw, 1, 0, 1, 2)
        self.exp.body_l.addWidget(chk_shift, 2, 0, 1, 2)

        lay.addWidget(self.exp, r, 0, 1, 4)
        r += 1

        return box

    def _build_preview(self, parent: QWidget) -> QGroupBox:
        box = QGroupBox("Preview", parent)
        lay = QVBoxLayout(box)
        lay.setContentsMargins(10, 10, 10, 10)
        lay.setSpacing(10)

        self.lbl_note = QLabel("", box)
        self.lbl_note.setWordWrap(True)
        self.lbl_note.setObjectName("crsNote")

        self._preview_style(box)
        lay.addWidget(self.lbl_note, 0)

        self.row_src = self._kv_row(box, "Source EPSG")
        self.row_dst = self._kv_row(box, "Target EPSG")
        self.row_eff = self._kv_row(box, "Strategy")
        
        self.v_src = self.row_src["v"]
        self.v_dst = self.row_dst["v"]
        self.v_eff = self.row_eff["v"]
        
        lay.addWidget(self.row_src["w"], 0)
        lay.addWidget(self.row_dst["w"], 0)
        lay.addWidget(self.row_eff["w"], 0)
        
        self._attach_copy_btn(
            self.row_src,
            get_text=lambda: self.row_src["v"].text(),
            tip="Copy source EPSG",
        )
        self._attach_copy_btn(
            self.row_dst,
            get_text=lambda: self.row_dst["v"].text(),
            tip="Copy target EPSG",
        )

        self.badge(
            "status",
            text="OK",
            accent="ok",
            tip="CRS consistency",
        )
        # UTM helper (zone + hemisphere)
        self.grp_utm = QGroupBox("Recommended UTM EPSG", box)
        u = QGridLayout(self.grp_utm)
        u.setContentsMargins(10, 10, 10, 10)
        u.setHorizontalSpacing(10)
        u.setVerticalSpacing(8)
        
        self.sp_zone = QSpinBox(self.grp_utm)
        self.sp_zone.setRange(1, 60)
        
        self.cmb_hemi = QComboBox(self.grp_utm)
        self.cmb_hemi.addItem("North", "north")
        self.cmb_hemi.addItem("South", "south")
        
        self.lbl_rec = QLabel("-", self.grp_utm)
        self.lbl_rec.setObjectName("crsVal")
        self.lbl_rec.setTextInteractionFlags(
            Qt.TextSelectableByMouse
        )
        
        self.btn_copy_rec = QPushButton("Copy", self.grp_utm)
        self.btn_copy_rec.setCursor(Qt.PointingHandCursor)
        self.btn_copy_rec.setFixedWidth(54)
        
        u.addWidget(QLabel("Zone:", self.grp_utm), 0, 0)
        u.addWidget(self.sp_zone, 0, 1)
        u.addWidget(QLabel("Hemisphere:", self.grp_utm), 0, 2)
        u.addWidget(self.cmb_hemi, 0, 3)
        
        u.addWidget(QLabel("EPSG:", self.grp_utm), 1, 0)
        u.addWidget(self.lbl_rec, 1, 1, 1, 2)
        u.addWidget(self.btn_copy_rec, 1, 3)
        
        lay.addWidget(self.grp_utm, 0)
        self.btn_use_rec = QPushButton("Use", self.grp_utm)
        self.btn_use_rec.setCursor(Qt.PointingHandCursor)
        self.btn_use_rec.setFixedWidth(46)
        
        self.btn_use_rec.setToolTip("Fill Target EPSG with this value")
        self.btn_copy_rec.setToolTip("Copy EPSG to clipboard")

        u.addWidget(self.btn_use_rec, 1, 4)


        return box

    def _preview_style(self, parent: QWidget) -> None:
        parent.setStyleSheet(
            "\n".join(
                [
                    "QLabel#crsNote {",
                    "  color: rgba(30,30,30,0.72);",
                    "  font-size: 11px;",
                    "}",
                    "QLabel#crsKey {",
                    "  color: rgba(30,30,30,0.72);",
                    "  font-size: 11px;",
                    "}",
                    "QLabel#crsVal {",
                    "  font-weight: 600;",
                    "  padding: 3px 10px;",
                    "  border-radius: 12px;",
                    "  border: 1px solid",
                    "    rgba(0,0,0,0.12);",
                    "  background: rgba(0,0,0,0.03);",
                    "}",
                ]
            )
        )
        
    def _attach_copy_btn(
        self,
        row: dict,
        *,
        get_text,
        tip: str,
    ) -> None:
        btn = QPushButton("Copy", row["w"])
        btn.setCursor(Qt.PointingHandCursor)
        btn.setToolTip(str(tip))
        btn.setFixedWidth(54)
    
        lay = row["w"].layout()
        if lay is None:
            return
        lay.addWidget(btn, 0)
    
        def _do() -> None:
            txt = str(get_text() or "").strip()
            if not txt or txt == "—":
                return
            self._copy_text(txt)
    
        btn.clicked.connect(_do)

    def _kv_row(self, parent: QWidget, key: str) -> dict:
        w = QWidget(parent)
        lay = QHBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(8)

        k = QLabel(f"{key}:", w)
        k.setObjectName("crsKey")

        v = QLabel("-", w)
        v.setObjectName("crsVal")
        v.setTextInteractionFlags(
            Qt.TextSelectableByMouse
        )

        lay.addWidget(k, 0)
        lay.addWidget(v, 1)
        return {"w": w, "v": v}

    def _lab(self, key: str, parent: QWidget) -> QLabel:
        s = self._fs[key]
        w = QLabel(f"{s.label}:", parent)
        if s.tooltip:
            w.setToolTip(s.tooltip)
        return w

    # -----------------------------------------------------------------
    # Wiring / refresh
    # -----------------------------------------------------------------
    def _wire(self) -> None:
        self.cmb_mode.currentIndexChanged.connect(self._update)
        self.sp_src.valueChanged.connect(self._update)
        self.sp_utm.valueChanged.connect(self._update)

        self.store.config_changed.connect(
            lambda _k: self.refresh(),
        )
        self.store.config_replaced.connect(
            lambda _cfg: self.refresh(),
        )
        self.sp_zone.valueChanged.connect(self._update)
        self.cmb_hemi.currentIndexChanged.connect(self._update)
            
        self.btn_copy_rec.clicked.connect(
            self._on_copy_recommended,
        )
        self.btn_use_rec.clicked.connect(self._use_recommended)

    def _use_recommended(self) -> None:
        txt = str(self.lbl_rec.text() or "").strip()
        if not txt or txt == "—":
            return
        try:
            self.store.patch({"utm_epsg": int(txt)})
        except Exception:
            return

    def _on_copy_recommended(self) -> None:
        txt = str(self.lbl_rec.text() or "").strip()
        if not txt or txt == "—":
            return
        self._copy_text(txt)

    def _update(self) -> None:
        self._sync_preview()

    def refresh(self) -> None:
        self._sync_preview()

    def _sync_preview(self) -> None:
        src = int(self.sp_src.value())
        dst = int(self.sp_utm.value())

        mode = str(self.cmb_mode.currentData() or "")
        mode = mode.strip()

        if not mode:
            mode = str(self.cmb_mode.currentText() or "")
            mode = mode.strip()

        self.v_src.setText(str(src) if src else "—")
        self.v_dst.setText(str(dst) if dst else "—")
        zone = int(self.sp_zone.value())
        hemi = str(self.cmb_hemi.currentData() or "")
        epsg = self._utm_epsg_from_zone(zone, hemi)
        
        self.lbl_rec.setText(str(epsg) if epsg else "—")
        self.btn_copy_rec.setEnabled(bool(epsg))
        self.btn_use_rec.setEnabled(bool(epsg))


        eff = self._describe_mode(mode)
        self.v_eff.setText(eff)

        ok = True
        msg = (
            "Choose a coordinate strategy and EPSG codes "
            "to standardize modelling coordinates."
        )

        if mode and "utm" in mode.lower() and not dst:
            ok = False
            msg = (
                "UTM conversion requires a target UTM EPSG."
            )

        if not src and not dst:
            msg = (
                "Tip: set source EPSG if your data are not "
                "already in a known CRS."
            )

        hint = "UTM: 326xx = north, 327xx = south."
        show_hint = bool(epsg) or ("utm" in mode.lower())
        
        if show_hint:
            msg = msg + "\n" + hint
        
        self.lbl_note.setText(msg)

        self.badge(
            "status",
            text="OK" if ok else "Check",
            accent="ok" if ok else "warn",
            tip=msg,
        )
        self.btn_copy_rec.setEnabled(bool(epsg))


    def _utm_epsg_from_zone(
        self,
        zone: int,
        hemi: str,
    ) -> int:
        z = int(zone)
        if z < 1 or z > 60:
            return 0
    
        h = (hemi or "").strip().lower()
        base = 32600 if h.startswith("n") else 32700
        return int(base + z)
    
    
    def _copy_text(self, text: str) -> None:
        try:
            QApplication.clipboard().setText(str(text))
        except Exception:
            return

    def _describe_mode(self, mode: str) -> str:
        m = (mode or "").strip().lower()

        if not m:
            return "Auto"

        if "utm" in m:
            return "Convert to UTM grid"

        if "raw" in m:
            return "Use raw lon/lat"

        if "project" in m or "proj" in m:
            return "Project to planar CRS"

        return str(mode)
