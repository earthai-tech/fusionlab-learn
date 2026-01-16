# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>

"""
Physics & constraints card (Setup tab).

Goals
-----
- Modern, compact "at-a-glance" physics setup.
- Keep core switches/weights inline.
- Delegate heavy edits to existing dialogs:
  - PhysicsConfigDialog
  - ScalarsLossDialog
"""

from __future__ import annotations

from typing import Any, Optional

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from .base import CardBase
from ..bindings import Binder

from ....config.store import GeoConfigStore
from ....config.prior_schema import (
    CHOICE_SPECS,
    FieldKey,
    PHYSICS_SCHEMA,
)

from ....dialogs.phys_dialogs import PhysicsConfigDialog
from ....dialogs.scalars_loss_dialog import ScalarsLossDialog


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
        self.btn.setText(title)
        self.btn.setCheckable(True)
        self.btn.setChecked(False)
        self.btn.setArrowType(Qt.RightArrow)
        self.btn.setToolButtonStyle(
            Qt.ToolButtonTextBesideIcon
        )

        self.body = QWidget(self)
        self.body.setVisible(False)
        self.body_lay = QGridLayout(self.body)
        self.body_lay.setContentsMargins(8, 6, 8, 6)
        self.body_lay.setHorizontalSpacing(10)
        self.body_lay.setVerticalSpacing(6)

        root.addWidget(self.btn, 0)
        root.addWidget(self.body, 0)

        self.btn.toggled.connect(self._toggle)

    def _toggle(self, on: bool) -> None:
        self.body.setVisible(bool(on))
        self.btn.setArrowType(
            Qt.DownArrow if on else Qt.RightArrow
        )


class _RangeEditor(QWidget):
    """
    Minimal range editor compatible with ScalarsLossDialog.

    Required API:
    - from_search_space_value(value, default)
    - to_search_space_value() -> Any
    """

    def __init__(
        self,
        *,
        min_allowed: float,
        max_allowed: float,
        decimals: int,
        show_sampling: bool,
        spin_width: int = 110,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)

        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)

        self.sp_min = QDoubleSpinBox(self)
        self.sp_max = QDoubleSpinBox(self)

        for sp in (self.sp_min, self.sp_max):
            sp.setDecimals(int(decimals))
            sp.setRange(float(min_allowed), float(max_allowed))
            sp.setFixedWidth(int(spin_width))

        self.sp_min.setToolTip("Min value")
        self.sp_max.setToolTip("Max value")

        self.cmb = QComboBox(self)
        self.cmb.addItem("linear", "linear")
        self.cmb.addItem("log", "log")
        self.cmb.setVisible(bool(show_sampling))
        self.cmb.setToolTip("Sampling (tuner)")

        lay.addWidget(QLabel("min", self), 0)
        lay.addWidget(self.sp_min, 0)
        lay.addWidget(QLabel("max", self), 0)
        lay.addWidget(self.sp_max, 0)
        lay.addWidget(self.cmb, 0)
        lay.addStretch(1)

    def from_search_space_value(
        self,
        value: Any,
        default: Any,
    ) -> None:
        v = default if value is None else value

        mn: float = 0.0
        mx: float = 0.0
        sampling: str = "linear"

        if isinstance(v, (int, float)):
            mn = float(v)
            mx = float(v)
        elif isinstance(v, dict):
            mn = float(v.get("min_value", 0.0))
            mx = float(v.get("max_value", mn))
            sampling = str(v.get("sampling", "linear"))
        else:
            mn = 0.0
            mx = 0.0

        self.sp_min.setValue(float(mn))
        self.sp_max.setValue(float(mx))

        idx = self.cmb.findData(sampling)
        if idx < 0:
            idx = self.cmb.findData("linear")
        if idx >= 0:
            self.cmb.setCurrentIndex(idx)

    def to_search_space_value(self) -> Any:
        mn = float(self.sp_min.value())
        mx = float(self.sp_max.value())

        sampling = "linear"
        if self.cmb.isVisible():
            sampling = str(self.cmb.currentData() or "linear")

        if mx < mn:
            mn, mx = mx, mn

        return {
            "type": "float",
            "min_value": mn,
            "max_value": mx,
            "sampling": sampling,
        }

class PhysicsConstraintsCard(CardBase):
    def __init__(
        self,
        *,
        store: GeoConfigStore,
        binder: Binder,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(
            section_id="physics",
            title="Physics & constraints",
            subtitle="PDE mode, bounds, and loss weights.",
            parent=parent,
        )
        self.store = store
        self.binder = binder

        self._lbl_state: Optional[QLabel] = None
        self._lbl_bounds: Optional[QLabel] = None
        self._lbl_tuner: Optional[QLabel] = None

        self._build()
        self._wire()

        self.refresh()

    # -----------------------------------------------------------------
    # Build
    # -----------------------------------------------------------------
    def _build(self) -> None:
        host = QWidget(self.body)
    
        root = QGridLayout(host)
        root.setContentsMargins(0, 0, 0, 0)
        root.setHorizontalSpacing(12)
        root.setVerticalSpacing(10)
    
        left = self._build_engine_box(host)
        right = self._build_weights_box(host)
        footer = self._build_footer_box(host)
    
        root.addWidget(left, 0, 0)
        root.addWidget(right, 0, 1)
        root.addWidget(footer, 1, 0, 1, 2)
    
        root.setColumnStretch(0, 1)
        root.setColumnStretch(1, 1)
    
        self.body_layout().addWidget(host, 0)


    def _build_engine_box(self, parent: QWidget) -> QWidget:
        box = QFrame(parent)
        box.setFrameShape(QFrame.NoFrame)

        lay = QGridLayout(box)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setHorizontalSpacing(10)
        lay.setVerticalSpacing(8)

        title = QLabel("Physics engine", box)
        title.setObjectName("secTitle")
        lay.addWidget(title, 0, 0, 1, 4)

        r = 1

        cmb_pde = QComboBox(box)
        self._bind_choice("pde_mode", cmb_pde)

        cmb_strat = QComboBox(box)
        self._bind_choice("training_strategy", cmb_strat)

        lay.addWidget(self._lab("pde_mode", box), r, 0)
        lay.addWidget(cmb_pde, r, 1)
        lay.addWidget(self._lab("training_strategy", box), r, 2)
        lay.addWidget(cmb_strat, r, 3)
        r += 1

        cmb_res = QComboBox(box)
        self._bind_choice("residual_method", cmb_res)

        cmb_bounds = QComboBox(box)
        self._bind_choice("bounds_mode", cmb_bounds)

        lay.addWidget(self._lab("residual_method", box), r, 0)
        lay.addWidget(cmb_res, r, 1)
        lay.addWidget(self._lab("bounds_mode", box), r, 2)
        lay.addWidget(cmb_bounds, r, 3)
        r += 1

        sp_warm = QSpinBox(box)
        sp_warm.setRange(0, 10_000_000)
        sp_ramp = QSpinBox(box)
        sp_ramp.setRange(0, 10_000_000)

        self.binder.bind_spin_box("physics_warmup_steps", sp_warm)
        self.binder.bind_spin_box("physics_ramp_steps", sp_ramp)

        lay.addWidget(self._lab("physics_warmup_steps", box), r, 0)
        lay.addWidget(sp_warm, r, 1)
        lay.addWidget(self._lab("physics_ramp_steps", box), r, 2)
        lay.addWidget(sp_ramp, r, 3)
        r += 1

        row = QWidget(box)
        rl = QHBoxLayout(row)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.setSpacing(10)

        chk_scale = QCheckBox("Scale PDE residuals", row)
        chk_subs = QCheckBox("Allow subs residual", row)
        chk_aux = QCheckBox("Track aux metrics", row)
        chk_dbg = QCheckBox("Debug physics grads", row)

        self.binder.bind_checkbox(
            "scale_pde_residuals",
            chk_scale,
        )
        self.binder.bind_checkbox(
            "allow_subs_residual",
            chk_subs,
        )
        self.binder.bind_checkbox(
            "track_aux_metrics",
            chk_aux,
        )
        self.binder.bind_checkbox(
            "debug_physics_grads",
            chk_dbg,
        )

        rl.addWidget(chk_scale)
        rl.addWidget(chk_subs)
        rl.addWidget(chk_aux)
        rl.addWidget(chk_dbg)
        rl.addStretch(1)

        lay.addWidget(row, r, 0, 1, 4)
        r += 1

        chk_clip = QCheckBox("Clip global norm", box)
        sp_clip = QDoubleSpinBox(box)
        sp_clip.setDecimals(6)
        sp_clip.setRange(0.0, 1e12)
        sp_clip.setSingleStep(0.1)

        self.binder.bind_optional_double_spin_box(
            "clip_global_norm",
            sp_clip,
            chk_clip,
        )

        lay.addWidget(QLabel("Grad clip:", box), r, 0)
        lay.addWidget(chk_clip, r, 1)
        lay.addWidget(sp_clip, r, 2, 1, 2)
        r += 1

        return box

    def _build_weights_box(self, parent: QWidget) -> QWidget:
        box = QFrame(parent)
        box.setFrameShape(QFrame.NoFrame)

        lay = QGridLayout(box)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setHorizontalSpacing(10)
        lay.setVerticalSpacing(8)

        title = QLabel("Loss weights", box)
        title.setObjectName("secTitle")
        lay.addWidget(title, 0, 0, 1, 4)

        def dspin(
            *,
            mn: float = 0.0,
            mx: float = 1e12,
            dec: int = 6,
            step: float = 1e-3,
        ) -> QDoubleSpinBox:
            sp = QDoubleSpinBox(box)
            sp.setRange(float(mn), float(mx))
            sp.setDecimals(int(dec))
            sp.setSingleStep(float(step))
            return sp

        r = 1

        sp_cons = dspin()
        sp_gw = dspin()
        sp_prior = dspin()
        sp_smooth = dspin()
        sp_bounds = dspin()
        sp_mv = dspin()
        sp_q = dspin()

        self.binder.bind_double_spin_box("lambda_cons", sp_cons)
        self.binder.bind_double_spin_box("lambda_gw", sp_gw)
        self.binder.bind_double_spin_box("lambda_prior", sp_prior)
        self.binder.bind_double_spin_box("lambda_smooth", sp_smooth)
        self.binder.bind_double_spin_box("lambda_bounds", sp_bounds)
        self.binder.bind_double_spin_box("lambda_mv", sp_mv)
        self.binder.bind_double_spin_box("lambda_q", sp_q)

        def add2(
            row: int,
            k1: str,
            w1: QWidget,
            k2: str,
            w2: QWidget,
        ) -> None:
            lay.addWidget(self._lab(k1, box), row, 0)
            lay.addWidget(w1, row, 1)
            lay.addWidget(self._lab(k2, box), row, 2)
            lay.addWidget(w2, row, 3)

        add2(r, "lambda_cons", sp_cons, "lambda_gw", sp_gw)
        r += 1
        add2(r, "lambda_prior", sp_prior, "lambda_smooth", sp_smooth)
        r += 1
        add2(r, "lambda_bounds", sp_bounds, "lambda_mv", sp_mv)
        r += 1
        cmb_base = self._baseline_combo(box)
        add2(
            r,
            "lambda_q",
            sp_q,
            "physics_baseline_mode",
            cmb_base,
        )
        r += 1

        exp = _Expander("Offsets & schedule", parent=box)
        lay.addWidget(exp, r, 0, 1, 4)
        r += 1

        self._fill_offset_controls(exp.body_lay, box)

        return box

    def _baseline_combo(self, parent: QWidget) -> QComboBox:
        cmb = QComboBox(parent)
        self._bind_choice("physics_baseline_mode", cmb)
        return cmb

    def _fill_offset_controls(
        self,
        lay: QGridLayout,
        parent: QWidget,
    ) -> None:
        cmb_mode = QComboBox(parent)
        self._bind_choice("offset_mode", cmb_mode)

        sp_off = QDoubleSpinBox(parent)
        sp_off.setDecimals(6)
        sp_off.setRange(0.0, 1e12)
        sp_off.setSingleStep(0.1)
        self.binder.bind_double_spin_box("lambda_offset", sp_off)

        chk_sched = QCheckBox("Use scheduler", parent)
        self.binder.bind_checkbox(
            "use_lambda_offset_scheduler",
            chk_sched,
        )

        sp_warm = QSpinBox(parent)
        sp_warm.setRange(0, 10_000_000)
        self.binder.bind_spin_box("lambda_offset_warmup", sp_warm)

        sp_s = QDoubleSpinBox(parent)
        sp_s.setDecimals(6)
        sp_s.setRange(0.0, 1e12)
        sp_s.setSingleStep(0.1)
        self.binder.bind_double_spin_box("lambda_offset_start", sp_s)

        sp_e = QDoubleSpinBox(parent)
        sp_e.setDecimals(6)
        sp_e.setRange(0.0, 1e12)
        sp_e.setSingleStep(0.1)
        self.binder.bind_double_spin_box("lambda_offset_end", sp_e)

        lay.addWidget(self._lab("offset_mode", parent), 0, 0)
        lay.addWidget(cmb_mode, 0, 1)
        lay.addWidget(self._lab("lambda_offset", parent), 0, 2)
        lay.addWidget(sp_off, 0, 3)

        lay.addWidget(chk_sched, 1, 0, 1, 2)
        lay.addWidget(self._lab("lambda_offset_warmup", parent), 1, 2)
        lay.addWidget(sp_warm, 1, 3)

        lay.addWidget(self._lab("lambda_offset_start", parent), 2, 0)
        lay.addWidget(sp_s, 2, 1)
        lay.addWidget(self._lab("lambda_offset_end", parent), 2, 2)
        lay.addWidget(sp_e, 2, 3)

    def _build_footer_box(self, parent: QWidget) -> QWidget:
        box = QFrame(parent)
        box.setObjectName("cfgFooter")

        lay = QGridLayout(box)
        lay.setContentsMargins(10, 10, 10, 10)
        lay.setHorizontalSpacing(10)
        lay.setVerticalSpacing(8)

        self._lbl_state = QLabel(box)
        self._lbl_state.setWordWrap(True)
        self._lbl_state.setObjectName("muted")

        self._lbl_bounds = QLabel(box)
        self._lbl_bounds.setWordWrap(True)
        self._lbl_bounds.setObjectName("muted")

        self._lbl_tuner = QLabel(box)
        self._lbl_tuner.setWordWrap(True)
        self._lbl_tuner.setObjectName("muted")

        lay.addWidget(QLabel("Status:", box), 0, 0)
        lay.addWidget(self._lbl_state, 0, 1, 1, 3)

        lay.addWidget(QLabel("Bounds:", box), 1, 0)
        lay.addWidget(self._lbl_bounds, 1, 1, 1, 3)

        lay.addWidget(QLabel("Tuning:", box), 2, 0)
        lay.addWidget(self._lbl_tuner, 2, 1, 1, 3)

        btn_row = QWidget(box)
        bl = QHBoxLayout(btn_row)
        bl.setContentsMargins(0, 0, 0, 0)
        bl.setSpacing(8)

        btn_phys = QPushButton("Edit physics…", btn_row)
        btn_phys.setObjectName("miniAction")
        btn_phys.clicked.connect(self._open_physics_dialog)

        btn_scal = QPushButton("Scalars && losses…", btn_row)
        btn_scal.setObjectName("miniAction")
        btn_scal.clicked.connect(self._open_scalars_dialog)

        bl.addWidget(btn_phys)
        bl.addWidget(btn_scal)
        bl.addStretch(1)

        lay.addWidget(btn_row, 3, 0, 1, 4)

        return box

    # -----------------------------------------------------------------
    # Wiring / refresh
    # -----------------------------------------------------------------
    def _wire(self) -> None:
        self.store.config_changed.connect(self._on_cfg_changed)
        self.store.config_replaced.connect(lambda _c: self.refresh())

    def _on_cfg_changed(self, _keys: object) -> None:
        self.refresh()

    def refresh(self) -> None:
        cfg = self.store.cfg

        pde = str(getattr(cfg, "pde_mode", "off") or "off")
        strat = str(getattr(cfg, "training_strategy", "") or "")
        res = str(getattr(cfg, "residual_method", "") or "")
        bnd = str(getattr(cfg, "bounds_mode", "") or "")

        if self._lbl_state is not None:
            if pde in {"off", "none"}:
                msg = (
                    "Physics is disabled "
                    f"(pde_mode={pde}). "
                    "Training uses data losses only."
                )
            else:
                msg = (
                    "Physics enabled "
                    f"(pde_mode={pde}, "
                    f"strategy={strat}, "
                    f"residuals={res})."
                )
            if bnd:
                msg = f"{msg} Bounds={bnd}."
            self._lbl_state.setText(msg)

        if self._lbl_bounds is not None:
            self._lbl_bounds.setText(self._bounds_text())

        if self._lbl_tuner is not None:
            self._lbl_tuner.setText(self._tuner_text())

    def _bounds_text(self) -> str:
        b = getattr(self.store.cfg, "physics_bounds", None)
        if not isinstance(b, dict) or not b:
            return "No physics_bounds configured."

        def fmt(k: str) -> str:
            v = b.get(k)
            if v is None:
                return "?"
            try:
                return f"{float(v):.3g}"
            except Exception:
                return str(v)

        parts = [
            f"K[{fmt('K_min')}, {fmt('K_max')}]",
            f"Ss[{fmt('Ss_min')}, {fmt('Ss_max')}]",
            f"tau[{fmt('tau_min')}, {fmt('tau_max')}]",
            f"H[{fmt('H_min')}, {fmt('H_max')}]",
        ]
        return "  ".join(parts)

    def _tuner_text(self) -> str:
        space = self.store.get("tuner_search_space", default={})
        if not isinstance(space, dict) or not space:
            return "Tuner search space uses defaults."

        keys = (
            "kappa_lr_mult",
            "mv_lr_mult",
            "init_mv",
            "init_kappa",
            "lambda_mv_reg",
        )
        hit = [k for k in keys if k in space]
        if not hit:
            return "Tuner search space: custom (other keys)."
        return "Tuner search space: custom (" + ", ".join(hit) + ")."

    # -----------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------
    def _spec(self, name: str) -> Optional[Any]:
        try:
            return PHYSICS_SCHEMA.get(FieldKey(name))
        except Exception:
            return None

    def _lab(self, key: str, parent: QWidget) -> QLabel:
        spec = self._spec(key)
    
        txt = key
        if spec is not None and getattr(spec, "label", None):
            txt = str(spec.label)
    
        lab = QLabel(f"{txt}:", parent)
    
        tip = ""
        if spec is not None:
            tip = str(getattr(spec, "tooltip", "") or "")
        if tip:
            lab.setToolTip(tip)
    
        return lab

    def _bind_choice(self, key: str, cmb: QComboBox) -> None:
        choices = CHOICE_SPECS.get(key, ())
        items = [(c, c) for c in choices]
        self.binder.bind_combo(
            key,
            cmb,
            items=items,
            editable=False,
            use_item_data=True,
        )

    # -----------------------------------------------------------------
    # Actions
    # -----------------------------------------------------------------
    def _open_physics_dialog(self) -> None:
        dlg = PhysicsConfigDialog(self, store=self.store)
        dlg.exec_()

    def _open_scalars_dialog(self) -> None:
        dlg = ScalarsLossDialog(
            parent=self,
            store=self.store,
            range_editor_cls=_RangeEditor,
        )
        dlg.exec_()
        self.refresh()
