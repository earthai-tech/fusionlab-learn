# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>

"""geoprior.ui.setup.cards.scaling

Scaling & units.

Modern UX goals
---------------
- Two-column layout: controls + preview.
- Stage-1 scaling toggles as "chips" with icons.
- Clear SI affine mapping (subsidence / head / misc).
- Compact "extras" area (stability + JSON kwargs).
- Store-driven (GeoConfigStore) with Binder for inputs.
"""

from __future__ import annotations

from typing import Optional

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QStyle,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ....config.store import GeoConfigStore
from ..bindings import Binder
from .base import CardBase

class _Chip(QToolButton):
    """Small checkable chip used for inline toggles."""

    def __init__(
        self,
        text: str,
        *,
        icon: QStyle.StandardPixmap,
        parent: QWidget,
    ) -> None:
        super().__init__(parent)

        self.setCheckable(True)
        self.setCursor(Qt.PointingHandCursor)
        self.setAutoRaise(True)
        self.setText(str(text))
        self.setToolButtonStyle(
            Qt.ToolButtonTextBesideIcon
        )
        self.setIcon(self.style().standardIcon(icon))

        # Local pill look (stable across themes)
        self.setStyleSheet(
            "\n".join(
                [
                    "QToolButton {",
                    "  padding: 4px 10px;",
                    "  border-radius: 12px;",
                    "  border: 1px solid",
                    "    rgba(0,0,0,0.14);",
                    "  background: rgba(0,0,0,0.02);",
                    "}",
                    "QToolButton:checked {",
                    "  border-color:",
                    "    rgba(46,49,145,0.40);",
                    "  background:",
                    "    rgba(46,49,145,0.10);",
                    "  font-weight: 600;",
                    "}",
                ]
            )
        )


class ScalingCard(CardBase):
    """Scaling & units (store-driven)."""

    def __init__(
        self,
        *,
        store: GeoConfigStore,
        binder: Binder,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(
            section_id="scaling",
            title="Scaling & units",
            subtitle=(
                "Control Stage-1 scaling and SI affine mapping "
                "for subsidence, head, and related fields."
            ),
            parent=parent,
        )

        self.store = store
        self.binder = binder

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

        self.grp_controls = self._build_controls(grid)
        self.grp_preview = self._build_preview(grid)

        g.addWidget(self.grp_controls, 0, 0)
        g.addWidget(self.grp_preview, 0, 1)

        body.addWidget(grid, 0)

        btn_copy = self.add_action(
            text="Copy",
            tip="Copy scaling summary to clipboard",
            icon=QStyle.SP_FileDialogDetailedView,
        )
        btn_copy.clicked.connect(self._copy_preview)

    def _build_controls(self, parent: QWidget) -> QGroupBox:
        box = QGroupBox("Controls", parent)
        root = QVBoxLayout(box)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(10)

        root.addWidget(self._build_stage1(box), 0)
        root.addWidget(self._build_policy(box), 0)
        root.addWidget(self._build_affine(box), 0)
        root.addWidget(self._build_extras(box), 0)

        return box

    def _build_stage1(self, parent: QWidget) -> QGroupBox:
        box = QGroupBox("Stage-1 scaling", parent)
        lay = QVBoxLayout(box)
        lay.setContentsMargins(10, 10, 10, 10)
        lay.setSpacing(8)

        row = QWidget(box)
        r = QHBoxLayout(row)
        r.setContentsMargins(0, 0, 0, 0)
        r.setSpacing(8)

        self.ch_h = _Chip(
            "H-field",
            icon=QStyle.SP_ArrowUp,
            parent=row,
        )
        self.ch_g = _Chip(
            "GWL",
            icon=QStyle.SP_ArrowDown,
            parent=row,
        )
        self.ch_z = _Chip(
            "z_surf",
            icon=QStyle.SP_ArrowRight,
            parent=row,
        )

        r.addWidget(self.ch_h, 0)
        r.addWidget(self.ch_g, 0)
        r.addWidget(self.ch_z, 0)
        r.addStretch(1)

        lay.addWidget(row, 0)

        # Subsidence kind (cumulative vs rate)
        form = QWidget(box)
        f = QGridLayout(form)
        f.setContentsMargins(0, 0, 0, 0)
        f.setHorizontalSpacing(10)
        f.setVerticalSpacing(8)
        f.setColumnStretch(1, 1)

        self.cmb_kind = QComboBox(form)
        self.binder.bind_combo(
            "subsidence_kind",
            self.cmb_kind,
            items=[
                ("cumulative", "cumulative"),
                ("rate", "rate"),
            ],
            editable=False,
            use_item_data=True,
        )

        f.addWidget(QLabel("Subsidence kind:", form), 0, 0)
        f.addWidget(self.cmb_kind, 0, 1)

        lay.addWidget(form, 0)
        return box

    def _build_policy(self, parent: QWidget) -> QGroupBox:
        box = QGroupBox("Units & policy", parent)
        lay = QGridLayout(box)
        lay.setContentsMargins(10, 10, 10, 10)
        lay.setHorizontalSpacing(10)
        lay.setVerticalSpacing(8)
        lay.setColumnStretch(1, 1)
        lay.setColumnStretch(3, 1)

        self.cmb_time = QComboBox(box)
        self.binder.bind_combo(
            "time_units",
            self.cmb_time,
            items=[
                ("year", "year"),
                ("day", "day"),
                ("second", "second"),
            ],
            editable=False,
            use_item_data=True,
        )

        self.cmb_policy = QComboBox(box)
        self.binder.bind_combo(
            "scaling_error_policy",
            self.cmb_policy,
            items=[
                ("raise", "raise"),
                ("warn", "warn"),
                ("ignore", "ignore"),
            ],
            editable=False,
            use_item_data=True,
        )

        self.chk_auto = QPushButton("Auto SI from Stage-1", box)
        self.chk_auto.setCheckable(True)
        self.chk_auto.setCursor(Qt.PointingHandCursor)
        self.chk_auto.setToolTip(
            "Use Stage-1 inferred affine mapping when available."
        )
        self.chk_auto.setStyleSheet(
            "\n".join(
                [
                    "QPushButton {",
                    "  padding: 5px 10px;",
                    "  border-radius: 12px;",
                    "}",
                ]
            )
        )
        self._bind_toggle_btn(
            "auto_si_affine_from_stage1",
            self.chk_auto,
        )

        lay.addWidget(QLabel("Time units:", box), 0, 0)
        lay.addWidget(self.cmb_time, 0, 1)
        lay.addWidget(QLabel("Error policy:", box), 0, 2)
        lay.addWidget(self.cmb_policy, 0, 3)

        lay.addWidget(self.chk_auto, 1, 0, 1, 4)
        return box

    def _build_affine(self, parent: QWidget) -> QGroupBox:
        box = QGroupBox("SI affine mapping", parent)
        lay = QGridLayout(box)
        lay.setContentsMargins(10, 10, 10, 10)
        lay.setHorizontalSpacing(10)
        lay.setVerticalSpacing(8)
        lay.setColumnStretch(1, 1)
        lay.setColumnStretch(3, 1)

        r = 0

        # Subsidence: unit, scale, bias
        self.sp_sub_u = self._dspin(
            box,
            minimum=0.0,
            maximum=1e12,
            decimals=9,
            step=1e-3,
        )
        self.sp_sub_s = self._dspin(
            box,
            minimum=-1e12,
            maximum=1e12,
            decimals=9,
            step=1e-3,
        )
        self.sp_sub_b = self._dspin(
            box,
            minimum=-1e12,
            maximum=1e12,
            decimals=6,
            step=1e-2,
        )

        self.binder.bind_double_spin_box(
            "subs_unit_to_si",
            self.sp_sub_u,
        )
        self.binder.bind_double_spin_box(
            "subs_scale_si",
            self.sp_sub_s,
        )
        self.binder.bind_double_spin_box(
            "subs_bias_si",
            self.sp_sub_b,
        )

        lay.addWidget(QLabel("Subs unit → SI:", box), r, 0)
        lay.addWidget(self.sp_sub_u, r, 1)
        lay.addWidget(QLabel("Subs scale (SI):", box), r, 2)
        lay.addWidget(self.sp_sub_s, r, 3)
        r += 1

        lay.addWidget(QLabel("Subs bias (SI):", box), r, 0)
        lay.addWidget(self.sp_sub_b, r, 1)
        r += 1

        # Head: unit + optional scale/bias
        self.sp_head_u = self._dspin(
            box,
            minimum=0.0,
            maximum=1e12,
            decimals=9,
            step=1e-3,
        )
        self.binder.bind_double_spin_box(
            "head_unit_to_si",
            self.sp_head_u,
        )

        self.en_head_s = QPushButton("Scale", box)
        self.en_head_s.setCheckable(True)
        self.en_head_s.setCursor(Qt.PointingHandCursor)

        self.en_head_b = QPushButton("Bias", box)
        self.en_head_b.setCheckable(True)
        self.en_head_b.setCursor(Qt.PointingHandCursor)

        self.sp_head_s = self._dspin(
            box,
            minimum=-1e12,
            maximum=1e12,
            decimals=9,
            step=1e-3,
        )
        self.sp_head_b = self._dspin(
            box,
            minimum=-1e12,
            maximum=1e12,
            decimals=6,
            step=1e-2,
        )

        self._style_toggle_btn(self.en_head_s)
        self._style_toggle_btn(self.en_head_b)

        self.binder.bind_optional_double_spin_box(
            "head_scale_si",
            self.sp_head_s,
            self._as_checkbox(self.en_head_s),
        )
        self.binder.bind_optional_double_spin_box(
            "head_bias_si",
            self.sp_head_b,
            self._as_checkbox(self.en_head_b),
        )

        row_scale = QWidget(box)
        rs = QHBoxLayout(row_scale)
        rs.setContentsMargins(0, 0, 0, 0)
        rs.setSpacing(6)
        rs.addWidget(self.en_head_s, 0)
        rs.addWidget(self.sp_head_s, 1)

        row_bias = QWidget(box)
        rb = QHBoxLayout(row_bias)
        rb.setContentsMargins(0, 0, 0, 0)
        rb.setSpacing(6)
        rb.addWidget(self.en_head_b, 0)
        rb.addWidget(self.sp_head_b, 1)

        lay.addWidget(QLabel("Head unit → SI:", box), r, 0)
        lay.addWidget(self.sp_head_u, r, 1)
        lay.addWidget(QLabel("Head scale:", box), r, 2)
        lay.addWidget(row_scale, r, 3)
        r += 1

        lay.addWidget(QLabel("Head bias:", box), r, 2)
        lay.addWidget(row_bias, r, 3)
        r += 1

        # Misc units (thickness, z_surf)
        self.sp_th_u = self._dspin(
            box,
            minimum=0.0,
            maximum=1e12,
            decimals=9,
            step=1e-3,
        )
        self.sp_z_u = self._dspin(
            box,
            minimum=0.0,
            maximum=1e12,
            decimals=9,
            step=1e-3,
        )

        self.binder.bind_double_spin_box(
            "thickness_unit_to_si",
            self.sp_th_u,
        )
        self.binder.bind_double_spin_box(
            "z_surf_unit_to_si",
            self.sp_z_u,
        )

        lay.addWidget(QLabel("Thickness unit → SI:", box), r, 0)
        lay.addWidget(self.sp_th_u, r, 1)
        lay.addWidget(QLabel("z_surf unit → SI:", box), r, 2)
        lay.addWidget(self.sp_z_u, r, 3)
        r += 1

        return box

    def _build_extras(self, parent: QWidget) -> QGroupBox:
        box = QGroupBox("Stability & extras", parent)
        lay = QGridLayout(box)
        lay.setContentsMargins(10, 10, 10, 10)
        lay.setHorizontalSpacing(10)
        lay.setVerticalSpacing(8)
        lay.setColumnStretch(1, 1)
        lay.setColumnStretch(3, 1)

        self.sp_hmin = self._dspin(
            box,
            minimum=0.0,
            maximum=1e9,
            decimals=6,
            step=1e-3,
        )
        self.binder.bind_double_spin_box(
            "h_field_min_si",
            self.sp_hmin,
        )

        self.sp_dt = self._dspin(
            box,
            minimum=0.0,
            maximum=1e9,
            decimals=9,
            step=1e-6,
        )
        self.binder.bind_double_spin_box(
            "dt_min_units",
            self.sp_dt,
        )

        lay.addWidget(QLabel("H-field min (SI):", box), 0, 0)
        lay.addWidget(self.sp_hmin, 0, 1)
        lay.addWidget(QLabel("dt_min_units:", box), 0, 2)
        lay.addWidget(self.sp_dt, 0, 3)

        self.ed_json = QLineEdit(box)
        self.ed_json.setPlaceholderText(
            "Optional scaling_kwargs.json"
        )
        self.binder.bind_line_edit(
            "scaling_kwargs_json_path",
            self.ed_json,
            on="editingFinished",
        )

        self.btn_browse = QToolButton(box)
        self.btn_browse.setCursor(Qt.PointingHandCursor)
        self.btn_browse.setIcon(
            self.style().standardIcon(
                QStyle.SP_DialogOpenButton
            )
        )
        self.btn_browse.setToolTip("Browse JSON file")

        self.btn_clear = QToolButton(box)
        self.btn_clear.setCursor(Qt.PointingHandCursor)
        self.btn_clear.setIcon(
            self.style().standardIcon(
                QStyle.SP_DialogResetButton
            )
        )
        self.btn_clear.setToolTip("Clear JSON path")

        row = QWidget(box)
        r = QHBoxLayout(row)
        r.setContentsMargins(0, 0, 0, 0)
        r.setSpacing(6)
        r.addWidget(self.ed_json, 1)
        r.addWidget(self.btn_browse, 0)
        r.addWidget(self.btn_clear, 0)

        lay.addWidget(QLabel("Scaling kwargs JSON:", box), 1, 0)
        lay.addWidget(row, 1, 1, 1, 3)

        return box

    def _build_preview(self, parent: QWidget) -> QGroupBox:
        box = QGroupBox("Preview", parent)
        lay = QVBoxLayout(box)
        lay.setContentsMargins(10, 10, 10, 10)
        lay.setSpacing(10)

        self.lbl_note = QLabel("", box)
        self.lbl_note.setWordWrap(True)
        self.lbl_note.setObjectName("scNote")

        self._preview_style(box)
        lay.addWidget(self.lbl_note, 0)

        self.row_s1 = self._kv_row(box, "Stage-1")
        self.row_pol = self._kv_row(box, "Policy")
        self.row_sub = self._kv_row(box, "Subsidence")
        self.row_head = self._kv_row(box, "Head")
        self.row_misc = self._kv_row(box, "Other units")
        self.row_json = self._kv_row(box, "JSON kwargs")

        for row in (
            self.row_s1,
            self.row_pol,
            self.row_sub,
            self.row_head,
            self.row_misc,
            self.row_json,
        ):
            lay.addWidget(row["w"], 0)

        self._attach_copy_btn(
            self.row_sub,
            tip="Copy subsidence formula",
        )
        self._attach_copy_btn(
            self.row_head,
            tip="Copy head formula",
        )

        self.badge(
            "status",
            text="OK",
            accent="ok",
            tip="Scaling consistency",
        )

        return box

    # -----------------------------------------------------------------
    # Wiring / refresh
    # -----------------------------------------------------------------
    def _wire(self) -> None:
        self.store.config_changed.connect(
            lambda _k: self.refresh(),
        )
        self.store.config_replaced.connect(
            lambda _cfg: self.refresh(),
        )

        for w in (
            self.ch_h,
            self.ch_g,
            self.ch_z,
        ):
            w.toggled.connect(self._on_stage1_toggle)

        self.cmb_kind.currentIndexChanged.connect(self.refresh)
        self.cmb_time.currentIndexChanged.connect(self.refresh)
        self.cmb_policy.currentIndexChanged.connect(self.refresh)

        for sp in (
            self.sp_sub_u,
            self.sp_sub_s,
            self.sp_sub_b,
            self.sp_head_u,
            self.sp_head_s,
            self.sp_head_b,
            self.sp_th_u,
            self.sp_z_u,
            self.sp_hmin,
            self.sp_dt,
        ):
            sp.valueChanged.connect(self.refresh)

        self.chk_auto.clicked.connect(self.refresh)
        self.en_head_s.clicked.connect(self.refresh)
        self.en_head_b.clicked.connect(self.refresh)

        self.btn_browse.clicked.connect(self._browse_json)
        self.btn_clear.clicked.connect(self._clear_json)

    def refresh(self) -> None:
        self._sync_stage1_chips()
        self._sync_preview()

    # -----------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------
    def _dspin(
        self,
        parent: QWidget,
        *,
        minimum: float,
        maximum: float,
        decimals: int,
        step: float,
    ) -> QDoubleSpinBox:
        sp = QDoubleSpinBox(parent)
        sp.setRange(float(minimum), float(maximum))
        sp.setDecimals(int(decimals))
        sp.setSingleStep(float(step))
        return sp

    def _sync_stage1_chips(self) -> None:
        cfg = self.store.cfg

        self._set_chip(self.ch_h, bool(cfg.scale_h_field))
        self._set_chip(self.ch_g, bool(cfg.scale_gwl))
        self._set_chip(self.ch_z, bool(cfg.scale_z_surf))

    def _set_chip(self, chip: QToolButton, on: bool) -> None:
        chip.blockSignals(True)
        chip.setChecked(bool(on))
        chip.blockSignals(False)

    def _on_stage1_toggle(self, _on: bool) -> None:
        patch = {
            "scale_h_field": bool(self.ch_h.isChecked()),
            "scale_gwl": bool(self.ch_g.isChecked()),
            "scale_z_surf": bool(self.ch_z.isChecked()),
        }
        self.store.patch(patch)

    def _sync_preview(self) -> None:
        cfg = self.store.cfg

        s1 = []
        if cfg.scale_h_field:
            s1.append("H-field")
        if cfg.scale_gwl:
            s1.append("GWL")
        if cfg.scale_z_surf:
            s1.append("z_surf")
        s1_txt = ", ".join(s1) if s1 else "off"

        pol = (
            f"time={cfg.time_units}, "
            f"policy={cfg.scaling_error_policy}"
        )

        subs = (
            f"si = (raw * {cfg.subs_unit_to_si:g}) "
            f"* {cfg.subs_scale_si:g} + {cfg.subs_bias_si:g}"
        )

        head = self._head_formula(
            unit=cfg.head_unit_to_si,
            scale=cfg.head_scale_si,
            bias=cfg.head_bias_si,
        )

        misc = (
            f"th={cfg.thickness_unit_to_si:g}, "
            f"z={cfg.z_surf_unit_to_si:g}"
        )

        j = (cfg.scaling_kwargs_json_path or "").strip()
        j_txt = j if j else "—"

        self.row_s1["v"].setText(s1_txt)
        self.row_pol["v"].setText(pol)
        self.row_sub["v"].setText(subs)
        self.row_head["v"].setText(head)
        self.row_misc["v"].setText(misc)
        self.row_json["v"].setText(j_txt)

        ok = True
        msg = (
            "Affine mapping converts dataset units to SI. "
            "Use Auto SI when Stage-1 inferred mapping exists."
        )

        if cfg.subs_unit_to_si <= 0 or cfg.head_unit_to_si <= 0:
            ok = False
            msg = "Unit→SI multipliers must be > 0."

        if cfg.scaling_error_policy == "ignore":
            msg = msg + "\nPolicy=ignore may hide issues."

        self.lbl_note.setText(msg)

        self.badge(
            "status",
            text="OK" if ok else "Check",
            accent="ok" if ok else "warn",
            tip=msg,
        )

        self.badge(
            "auto",
            text="auto" if cfg.auto_si_affine_from_stage1 else "manual",
            accent="ok" if cfg.auto_si_affine_from_stage1 else "warn",
            tip="Auto SI affine from Stage-1",
        )

    def _head_formula(
        self,
        *,
        unit: float,
        scale: Optional[float],
        bias: Optional[float],
    ) -> str:
        base = f"si = raw * {unit:g}"
        if scale is not None:
            base = base + f" * {float(scale):g}"
        if bias is not None:
            base = base + f" + {float(bias):g}"
        return base

    def _preview_style(self, parent: QWidget) -> None:
        parent.setStyleSheet(
            "\n".join(
                [
                    "QLabel#scNote {",
                    "  color: rgba(30,30,30,0.72);",
                    "  font-size: 11px;",
                    "}",
                    "QLabel#scKey {",
                    "  color: rgba(30,30,30,0.72);",
                    "  font-size: 11px;",
                    "}",
                    "QLabel#scVal {",
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

    def _kv_row(self, parent: QWidget, key: str) -> dict:
        w = QWidget(parent)
        lay = QHBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(8)

        k = QLabel(f"{key}:", w)
        k.setObjectName("scKey")

        v = QLabel("—", w)
        v.setObjectName("scVal")
        v.setTextInteractionFlags(
            Qt.TextSelectableByMouse
        )

        lay.addWidget(k, 0)
        lay.addWidget(v, 1)
        return {"w": w, "v": v}

    def _attach_copy_btn(self, row: dict, *, tip: str) -> None:
        btn = QPushButton("Copy", row["w"])
        btn.setCursor(Qt.PointingHandCursor)
        btn.setToolTip(str(tip))
        btn.setFixedWidth(54)

        lay = row["w"].layout()
        if lay is None:
            return
        lay.addWidget(btn, 0)

        def _do() -> None:
            txt = str(row["v"].text() or "").strip()
            if not txt or txt == "—":
                return
            self._copy_text(txt)

        btn.clicked.connect(_do)

    def _copy_preview(self) -> None:
        txt = self._collect_preview_text()
        self._copy_text(txt)

    def _collect_preview_text(self) -> str:
        rows = [
            f"Stage-1: {self.row_s1['v'].text()}",
            f"Policy : {self.row_pol['v'].text()}",
            f"Subs   : {self.row_sub['v'].text()}",
            f"Head   : {self.row_head['v'].text()}",
            f"Misc   : {self.row_misc['v'].text()}",
            f"JSON   : {self.row_json['v'].text()}",
        ]
        return "\n".join(rows)

    def _copy_text(self, text: str) -> None:
        try:
            QApplication.clipboard().setText(str(text))
        except Exception:
            return

    def _browse_json(self) -> None:
        path, _flt = QFileDialog.getOpenFileName(
            self,
            "Select scaling kwargs JSON",
            "",
            "JSON (*.json);;All files (*.*)",
        )
        if not path:
            return
        self.store.patch({"scaling_kwargs_json_path": path})

    def _clear_json(self) -> None:
        self.store.patch({"scaling_kwargs_json_path": None})

    # -----------------------------------------------------------------
    # Tiny helpers (toggle buttons + Binder optional)
    # -----------------------------------------------------------------
    def _style_toggle_btn(self, btn: QPushButton) -> None:
        btn.setStyleSheet(
            "\n".join(
                [
                    "QPushButton {",
                    "  padding: 4px 10px;",
                    "  border-radius: 12px;",
                    "  border: 1px solid rgba(0,0,0,0.16);",
                    "  background: rgba(0,0,0,0.02);",
                    "  color: rgba(30,30,30,0.88);",          # readable in light mode
                    "}",
                    "QPushButton:hover:enabled {",
                    "  background: rgba(46,49,145,0.08);",
                    "  border-color: rgba(46,49,145,0.28);",
                    "  color: rgba(46,49,145,0.95);",         # optional: brand on hover
                    "}",
                    "QPushButton:checked {",
                    "  border-color: rgba(46,49,145,0.40);",
                    "  background: rgba(46,49,145,0.10);",
                    "  font-weight: 600;",
                    "  color: rgba(46,49,145,0.98);",         # readable + consistent
                    "}",
                    "QPushButton:disabled {",
                    "  color: rgba(100,116,139,0.55);",
                    "}",
                ]
            )
        )


    def _bind_toggle_btn(self, key: str, btn: QPushButton) -> None:
        # QPushButton checkable → store boolean
        def _push(on: bool) -> None:
            self.store.patch({str(key): bool(on)})

        btn.toggled.connect(_push)

        def _pull() -> None:
            v = bool(getattr(self.store.cfg, str(key)))
            btn.blockSignals(True)
            btn.setChecked(v)
            btn.blockSignals(False)

        self.store.config_changed.connect(lambda _k: _pull())
        self.store.config_replaced.connect(lambda _c: _pull())
        _pull()

    def _as_checkbox(self, btn: QPushButton):
        """
        Adapter: Binder expects a QCheckBox for optional binds.

        We provide a tiny wrapper with the needed API.
        """
        class _Wrap:
            def __init__(self, b: QPushButton) -> None:
                self._b = b

            def setChecked(self, on: bool) -> None:
                self._b.setChecked(bool(on))

            def isChecked(self) -> bool:
                return bool(self._b.isChecked())

            @property
            def toggled(self):
                return self._b.toggled

        return _Wrap(btn)

