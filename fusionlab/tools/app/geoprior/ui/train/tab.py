# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple

from PyQt5.QtCore import Qt, QSignalBlocker, pyqtSignal, QSize
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
    QSizePolicy,
    QScrollArea,
    QFrame,
    QSplitter,
    QToolButton,
    QListWidget,
    QListWidgetItem,
)
from PyQt5.QtGui import QFontMetrics


from ...config.store import GeoConfigStore
from ...config.prior_schema import FieldKey
from ...device_options import runtime_summary_text

from .head import TrainHeadBar
from .run_preview import RunPreviewPanel


MakeCardFn = Callable[[str], Tuple[QWidget, QVBoxLayout]]
MakeRunBtnFn = Callable[[str], QToolButton]


__all__ = ["TrainTab"]

class TrainTab(QWidget):
    """
    Store-driven Train tab (v3.2).

    - No direct reads from app.geo_cfg.
    - Writes go to GeoConfigStore immediately.
    - UI refreshes on store changes.
    """

    run_clicked = pyqtSignal()
    advanced_clicked = pyqtSignal()
    features_clicked = pyqtSignal()
    arch_clicked = pyqtSignal()
    prob_clicked = pyqtSignal()
    physics_clicked = pyqtSignal()

    def __init__(
        self,
        *,
        store: GeoConfigStore,
        make_card: MakeCardFn,
        make_run_button: MakeRunBtnFn,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)

        self._store = store
        self._make_card = make_card
        self._make_run_button = make_run_button

        self._writing = False
        self._bindings: Dict[str, Callable[[], None]] = {}
        self._presets: Dict[str, Dict[str, Any]] = (
            self._build_presets()
        )
        self._nav_row_by_key: Dict[str, QWidget] = {}
        self._nav_chip_by_key: Dict[str, QLabel] = {}
        
        self._build_ui()
        self._wire_ui()
        self._wire_store()
        self.refresh_from_store()
        
    def _fmt_lr(self, v: float) -> str:
        try:
            fv = float(v)
        except Exception:
            return str(v)
        if fv == 0.0:
            return "0"
        if abs(fv) < 1e-3:
            return f"{fv:.1e}"
        return f"{fv:.6f}"


    def _add_nav_item(self, text: str, key: str) -> None:
        it = QListWidgetItem()
        it.setData(Qt.UserRole, key)
    
        row = QWidget()
        row.setObjectName("navRow")
        row.setProperty("selected", False)
        row.setAttribute(Qt.WA_StyledBackground, True)
        row.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Fixed,
        )
        row.setMinimumHeight(34)
    
        lay = QHBoxLayout(row)
        lay.setContentsMargins(10, 6, 14, 6)
        lay.setSpacing(8)
    
        lbl = QLabel(text)
        lbl.setObjectName("navText")
        lbl.setMinimumWidth(0)
        lbl.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Fixed,
        )
    
        chip = QLabel("OK")
        chip.setObjectName("navChip")
        chip.setProperty("status", "ok")
        chip.setAlignment(Qt.AlignCenter)
        chip.setMinimumHeight(18)
    
        lay.addWidget(lbl, 1)
        lay.addWidget(chip, 0, Qt.AlignRight)
    
        self.nav_list.addItem(it)
        self.nav_list.setItemWidget(it, row)
    
        # IMPORTANT: do not freeze width, only height
        h = row.sizeHint().height()
        it.setSizeHint(QSize(0, h))
    
        self._nav_row_by_key[key] = row
        self._nav_chip_by_key[key] = chip

        
    def _nav_apply_selection(self, row: int) -> None:
        for i in range(self.nav_list.count()):
            it = self.nav_list.item(i)
            if it is None:
                continue
            key = str(it.data(Qt.UserRole))
            w = self._nav_row_by_key.get(key)
            if w is None:
                continue
            w.setProperty("selected", i == row)
            w.style().unpolish(w)
            w.style().polish(w)
            w.update()
        
    def _set_nav_chip(
        self,
        key: str,
        status: str,
        text: str,
    ) -> None:
        chip = self._nav_chip_by_key.get(key)
        if chip is None:
            return
        chip.setText(text)
        chip.setProperty("status", status)
        chip.style().unpolish(chip)
        chip.style().polish(chip)
        chip.update()

    def _mk_disclosure(
        self,
        *,
        summary: QLabel,
        body: QWidget,
        expanded: bool = False,
    ) -> QToolButton:
        btn = QToolButton()
        btn.setObjectName("disclosure")
        btn.setCheckable(True)
        btn.setChecked(bool(expanded))
        btn.setToolButtonStyle(
            Qt.ToolButtonTextBesideIcon
        )
        btn.setText("Edit")
        btn.setArrowType(
            Qt.DownArrow if expanded else Qt.RightArrow
        )
        body.setVisible(bool(expanded))
    
        def _toggle(on: bool) -> None:
            body.setVisible(bool(on))
            btn.setArrowType(
                Qt.DownArrow if on else Qt.RightArrow
            )
            # keep scroll stable after expand/collapse
            p = body.parentWidget()
            if p is not None:
                p.updateGeometry()
    
        btn.toggled.connect(_toggle)
        summary.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Fixed,
        )
        return btn
        

    def _on_nav_changed(self, row: int) -> None:
        self._nav_apply_selection(row)
    
        it = self.nav_list.item(row)
        if it is None:
            return
    
        key = str(it.data(Qt.UserRole))
        target = {
            "temporal": self._card_temporal,
            "training": self._card_training,
            "physw": self._card_physw,
            "sched": self._card_sched,
            "advopt": self._card_advopt,
        }.get(key)
    
        if target is None:
            return
    
        if getattr(self, "_scroll", None) is not None:
            self._scroll.ensureWidgetVisible(target, 0, 16)


    def _refresh_summaries(self) -> None:
        if not hasattr(self, "lbl_sum_time"):
            return
    
        te = int(self.sp_train_end.value())
        fs = int(self.sp_forecast_start.value())
        fh = int(self.sp_forecast_horizon.value())
        ts = int(self.sp_time_steps.value())
    
        ep = int(self.sp_epochs.value())
        bs = int(self.sp_batch_size.value())
        lr = self._fmt_lr(self.sp_lr.value())
    
        mode = str(self.cmb_pde_mode.currentText())
        lc = float(self.sb_lcons.value())
        lg = float(self.sb_lgw.value())
        lp = float(self.sb_lprior.value())
    
        wu = int(self.sp_phys_warmup.value())
        rp = int(self.sp_phys_ramp.value())
        sc = bool(self.chk_scale_pde.isChecked())
        ev = bool(self.chk_eval_training.isChecked())
        bf = bool(self.chk_build_future.isChecked())
    
        self.lbl_sum_time.setText(
            f"end={te}  start={fs}  "
            f"h={fh}y  lookback={ts}"
        )
        self.lbl_sum_train.setText(
            f"epochs={ep}  batch={bs}  lr={lr}"
        )
        self.lbl_sum_phys.setText(
            f"mode={mode}  λc={lc:.2f}  "
            f"λgw={lg:.2f}  λp={lp:.2f}"
        )
        self.lbl_sum_sched.setText(
            f"warmup={wu}  ramp={rp}  "
            f"scale={int(sc)}"
        )
        self.lbl_sum_advopt.setText(
            f"eval={int(ev)}  "
            f"future_npz={int(bf)}"
        )
    
        plan_txt = (
            "Run plan\n"
            f"• Years: end={te}, start={fs}, "
            f"h={fh}y, lookback={ts}\n"
            f"• Train: epochs={ep}, "
            f"batch={bs}, lr={lr}\n"
            f"• Physics: mode={mode}, "
            f"warmup={wu}, ramp={rp}, "
            f"scale={int(sc)}\n"
            f"• Options: eval={int(ev)}, "
            f"future_npz={int(bf)}"
        )
    
        if hasattr(self, "run_preview"):
            self.run_preview.set_plan_text(plan_txt)
            self.run_preview.set_plan(
                epochs=ep,
                warmup=wu,
                ramp=rp,
                mode=mode,
                lambdas={
                    "c": lc,
                    "gw": lg,
                    "p": lp,
                    "s": float(self.sb_lsmooth.value()),
                    "mv": float(self.sb_lmv.value()),
                },
                scale=sc,
                eval_on=ev,
                future=bf,
            )
    
        if hasattr(self, "head"):
            self.head.set_plan_text(plan_txt)
    
        self._update_nav_chips()

    def _update_nav_chips(self) -> None:
        te = int(self.sp_train_end.value())
        fs = int(self.sp_forecast_start.value())
    
        if fs <= te:
            self._set_nav_chip("temporal", "warn", "Fix")
        else:
            self._set_nav_chip("temporal", "ok", "OK")
    
        ep = int(self.sp_epochs.value())
        lr = float(self.sp_lr.value())
        if ep <= 0 or lr <= 0.0:
            self._set_nav_chip("training", "err", "Err")
        else:
            self._set_nav_chip("training", "ok", "OK")
    
        mode = str(self.cmb_pde_mode.currentText())
        lp = float(self.sb_lprior.value())
        lc = float(self.sb_lcons.value())
        lg = float(self.sb_lgw.value())
    
        if mode == "off":
            self._set_nav_chip("physw", "off", "OFF")
        elif lp <= 0.0 and (lc > 0.0 or lg > 0.0):
            self._set_nav_chip("physw", "warn", "λp?")
        else:
            self._set_nav_chip("physw", "ok", "OK")
    
        wu = int(self.sp_phys_warmup.value())
        rp = int(self.sp_phys_ramp.value())
        if mode == "off" and (wu > 0 or rp > 0):
            self._set_nav_chip("sched", "warn", "Sched")
        else:
            self._set_nav_chip("sched", "ok", "OK")
    
        self._set_nav_chip("advopt", "ok", "OK")


    # -----------------------------------------------------------------
    # UI build
    # -----------------------------------------------------------------
    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(6, 6, 6, 6)
        outer.setSpacing(8)
    
        def _field(w: QWidget) -> None:
            w.setSizePolicy(
                QSizePolicy.Expanding,
                QSizePolicy.Fixed,
            )
            w.setMinimumHeight(26)
            w.setMinimumWidth(140)
    
        def _btn(b: QPushButton) -> None:
            b.setSizePolicy(
                QSizePolicy.Minimum,
                QSizePolicy.Fixed,
            )
            b.setMinimumHeight(30)
    
        # ==========================================
        # Main split: [LEFT] | [HEAD + WORKSPACE]
        # ==========================================
        main_split = QSplitter(Qt.Horizontal, self)
        main_split.setHandleWidth(6)
        main_split.setChildrenCollapsible(False)
        outer.addWidget(main_split, 1)
        
        # -----------------------------------------
        # Left column: [Setup checklist] + [Computer]
        # -----------------------------------------
        left_col = QWidget(self)
        left_l = QVBoxLayout(left_col)
        left_l.setContentsMargins(0, 0, 0, 0)
        left_l.setSpacing(8)
        
        # --------
        # Card 1: Setup checklist
        # --------
        nav = QFrame(left_col)
        nav.setObjectName("trainNavCard")
        nav_l = QVBoxLayout(nav)
        nav_l.setContentsMargins(10, 10, 10, 10)
        nav_l.setSpacing(10)
        
        nav_title = QLabel("Setup checklist")
        nav_title.setObjectName("trainNavTitle")
        nav_l.addWidget(nav_title)
        
        self.btn_features = QPushButton("Feature")
        self.btn_arch = QPushButton("Arch")
        self.btn_prob = QPushButton("Prob")
        self.physics_btn = QPushButton("Physics")
        
        for b in (
            self.btn_features,
            self.btn_arch,
            self.btn_prob,
            self.physics_btn,
        ):
            _btn(b)
            b.setObjectName("miniAction")
        
        acts = QWidget(nav)
        acts_l = QGridLayout(acts)
        acts_l.setContentsMargins(0, 0, 0, 0)
        acts_l.setHorizontalSpacing(8)
        acts_l.setVerticalSpacing(8)
        
        acts_l.addWidget(self.btn_features, 0, 0)
        acts_l.addWidget(self.btn_arch, 0, 1)
        acts_l.addWidget(self.btn_prob, 1, 0)
        acts_l.addWidget(self.physics_btn, 1, 1)
        nav_l.addWidget(acts)
        
        self.nav_list = QListWidget()
        self.nav_list.setObjectName("trainNavList")
        self.nav_list.setSpacing(2)
        self.nav_list.setUniformItemSizes(True)
        
        self._add_nav_item("Temporal window", "temporal")
        self._add_nav_item("Training", "training")
        self._add_nav_item("Physics weights", "physw")
        self._add_nav_item("Physics schedule", "sched")
        self._add_nav_item("Advanced", "advopt")
        
        # keep checklist compact (no big empty area)
        self.nav_list.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Fixed,
        )
        self._fix_nav_list_height()
        
        nav_l.addWidget(self.nav_list, 0)
        left_l.addWidget(nav, 0)
        
        # --------
        # Card 2: Computer details (fills the remaining space)
        # --------
        comp = QFrame(left_col)
        comp.setObjectName("trainNavCard")
        comp_l = QVBoxLayout(comp)
        comp_l.setContentsMargins(10, 10, 10, 10)
        comp_l.setSpacing(10)
        
        comp_title = QLabel("Computer details")
        comp_title.setObjectName("trainNavTitle")
        comp_l.addWidget(comp_title, 0)
        
        self.lbl_compute_nav = QLabel("")
        self.lbl_compute_nav.setObjectName("runComputeText")
        self.lbl_compute_nav.setWordWrap(True)
        self.lbl_compute_nav.setTextInteractionFlags(
            Qt.TextSelectableByMouse
        )
        comp_l.addWidget(self.lbl_compute_nav, 0)
        comp_l.addStretch(1)
        
        left_l.addWidget(comp, 1)
        
        main_split.addWidget(left_col)

        # -----------------------------------------
        # Right: Head + workspace (Editor|Preview)
        # -----------------------------------------
        right = QWidget(self)
        right_l = QVBoxLayout(right)
        right_l.setContentsMargins(0, 0, 0, 0)
        right_l.setSpacing(8)
    
        # Head pinned ONLY over the right workspace
        self.head = TrainHeadBar(store=self._store)
        self.head.set_presets(list(self._presets.keys()))
        self.head.refresh_from_store()
        right_l.addWidget(self.head, 0)
    
        # Inner split: [Editor scroll] | [Preview scroll]
        work_split = QSplitter(Qt.Horizontal, right)
        work_split.setHandleWidth(6)
        work_split.setChildrenCollapsible(False)
        right_l.addWidget(work_split, 1)
    
        # ----------------------------
        # Center: Editor (scroll)
        # ----------------------------
        center_scroll = QScrollArea(right)
        center_scroll.setWidgetResizable(True)
        center_scroll.setFrameShape(QFrame.NoFrame)
        self._scroll = center_scroll
    
        center_page = QWidget(center_scroll)
        center_scroll.setWidget(center_page)
    
        center_l = QVBoxLayout(center_page)
        center_l.setContentsMargins(0, 0, 0, 0)
        center_l.setSpacing(10)
    
        # ==================================================
        # Card: Temporal window
        # ==================================================
        temp_card, temp_box = self._make_card("Temporal window")
        self._card_temporal = temp_card
    
        self.lbl_sum_time = QLabel("")
        self.lbl_sum_time.setObjectName("sumLine")
    
        temp_body = QWidget(temp_card)
        temp_body_l = QVBoxLayout(temp_body)
        temp_body_l.setContentsMargins(0, 0, 0, 0)
        temp_body_l.setSpacing(6)
    
        self.sp_train_end = QSpinBox()
        self.sp_train_end.setRange(2000, 2100)
        _field(self.sp_train_end)
    
        self.sp_forecast_start = QSpinBox()
        self.sp_forecast_start.setRange(2000, 2100)
        _field(self.sp_forecast_start)
    
        self.sp_forecast_horizon = QSpinBox()
        self.sp_forecast_horizon.setRange(1, 50)
        _field(self.sp_forecast_horizon)
    
        self.sp_time_steps = QSpinBox()
        self.sp_time_steps.setRange(1, 50)
        _field(self.sp_time_steps)
    
        grid_t = QGridLayout()
        grid_t.setHorizontalSpacing(10)
        grid_t.setVerticalSpacing(8)
    
        grid_t.addWidget(QLabel("Train end year:"), 0, 0)
        grid_t.addWidget(self.sp_train_end, 0, 1)
        grid_t.addWidget(QLabel("Forecast start year:"), 1, 0)
        grid_t.addWidget(self.sp_forecast_start, 1, 1)
        grid_t.addWidget(
            QLabel("Forecast horizon (years):"),
            2,
            0,
        )
        grid_t.addWidget(self.sp_forecast_horizon, 2, 1)
        grid_t.addWidget(
            QLabel("Time steps (look-back):"),
            3,
            0,
        )
        grid_t.addWidget(self.sp_time_steps, 3, 1)
    
        grid_t.setColumnStretch(0, 0)
        grid_t.setColumnStretch(1, 1)
        temp_body_l.addLayout(grid_t)
    
        temp_hdr = QHBoxLayout()
        temp_hdr.setContentsMargins(0, 0, 0, 0)
        temp_hdr.setSpacing(8)
    
        temp_btn = self._mk_disclosure(
            summary=self.lbl_sum_time,
            body=temp_body,
            expanded=False,
        )
        temp_hdr.addWidget(self.lbl_sum_time, 1)
        temp_hdr.addWidget(temp_btn, 0)
    
        temp_box.addLayout(temp_hdr)
        temp_box.addWidget(temp_body)
    
        # ==================================================
        # Card: Training
        # ==================================================
        train_card, train_box = self._make_card("Training")
        self._card_training = train_card
    
        self.lbl_sum_train = QLabel("")
        self.lbl_sum_train.setObjectName("sumLine")
    
        train_body = QWidget(train_card)
        train_body_l = QVBoxLayout(train_body)
        train_body_l.setContentsMargins(0, 0, 0, 0)
        train_body_l.setSpacing(6)
    
        self.sp_epochs = QSpinBox()
        self.sp_epochs.setRange(1, 5000)
        _field(self.sp_epochs)
    
        self.sp_batch_size = QSpinBox()
        self.sp_batch_size.setRange(1, 1024)
        _field(self.sp_batch_size)
    
        self.sp_lr = QDoubleSpinBox()
        self.sp_lr.setDecimals(6)
        self.sp_lr.setRange(1e-6, 1e-2)
        self.sp_lr.setSingleStep(1e-5)
        _field(self.sp_lr)
    
        grid_tr = QGridLayout()
        grid_tr.setHorizontalSpacing(10)
        grid_tr.setVerticalSpacing(8)
    
        grid_tr.addWidget(QLabel("Epochs:"), 0, 0)
        grid_tr.addWidget(self.sp_epochs, 0, 1)
        grid_tr.addWidget(QLabel("Batch size:"), 1, 0)
        grid_tr.addWidget(self.sp_batch_size, 1, 1)
        grid_tr.addWidget(QLabel("Learning rate:"), 2, 0)
        grid_tr.addWidget(self.sp_lr, 2, 1)
    
        grid_tr.setColumnStretch(0, 0)
        grid_tr.setColumnStretch(1, 1)
        train_body_l.addLayout(grid_tr)
    
        train_hdr = QHBoxLayout()
        train_hdr.setContentsMargins(0, 0, 0, 0)
        train_hdr.setSpacing(8)
    
        train_btn = self._mk_disclosure(
            summary=self.lbl_sum_train,
            body=train_body,
            expanded=False,
        )
        train_hdr.addWidget(self.lbl_sum_train, 1)
        train_hdr.addWidget(train_btn, 0)
    
        train_box.addLayout(train_hdr)
        train_box.addWidget(train_body)
    
        # ==================================================
        # Card: Physics weights
        # ==================================================
        phys_card, phys_box = self._make_card("Physics weights")
        self._card_physw = phys_card
    
        self.lbl_sum_phys = QLabel("")
        self.lbl_sum_phys.setObjectName("sumLine")
    
        phys_body = QWidget(phys_card)
        phys_body_l = QVBoxLayout(phys_body)
        phys_body_l.setContentsMargins(0, 0, 0, 0)
        phys_body_l.setSpacing(6)
    
        self.cmb_pde_mode = QComboBox()
        self.cmb_pde_mode.addItems(
            ["both", "consolidation", "gw_flow", "off"]
        )
        _field(self.cmb_pde_mode)
    
        def _lambda_spin() -> QDoubleSpinBox:
            sb = QDoubleSpinBox()
            sb.setDecimals(4)
            sb.setRange(0.0, 10.0)
            sb.setSingleStep(0.005)
            _field(sb)
            return sb
    
        self.sb_lcons = _lambda_spin()
        self.sb_lgw = _lambda_spin()
        self.sb_lprior = _lambda_spin()
        self.sb_lsmooth = _lambda_spin()
        self.sb_lmv = _lambda_spin()
    
        grid_w = QGridLayout()
        grid_w.setHorizontalSpacing(10)
        grid_w.setVerticalSpacing(8)
    
        grid_w.addWidget(QLabel("PDE mode:"), 0, 0)
        grid_w.addWidget(self.cmb_pde_mode, 0, 1)
        grid_w.addWidget(QLabel("λ consolidation:"), 1, 0)
        grid_w.addWidget(self.sb_lcons, 1, 1)
        grid_w.addWidget(QLabel("λ GW flow:"), 2, 0)
        grid_w.addWidget(self.sb_lgw, 2, 1)
        grid_w.addWidget(QLabel("λ prior:"), 3, 0)
        grid_w.addWidget(self.sb_lprior, 3, 1)
        grid_w.addWidget(QLabel("λ smooth:"), 4, 0)
        grid_w.addWidget(self.sb_lsmooth, 4, 1)
        grid_w.addWidget(QLabel("λ m_v:"), 5, 0)
        grid_w.addWidget(self.sb_lmv, 5, 1)
    
        grid_w.setColumnStretch(0, 0)
        grid_w.setColumnStretch(1, 1)
        phys_body_l.addLayout(grid_w)
    
        phys_hdr = QHBoxLayout()
        phys_hdr.setContentsMargins(0, 0, 0, 0)
        phys_hdr.setSpacing(8)
    
        phys_btn = self._mk_disclosure(
            summary=self.lbl_sum_phys,
            body=phys_body,
            expanded=False,
        )
        phys_hdr.addWidget(self.lbl_sum_phys, 1)
        phys_hdr.addWidget(phys_btn, 0)
    
        phys_box.addLayout(phys_hdr)
        phys_box.addWidget(phys_body)
    
        # ==================================================
        # Card: Physics schedule
        # ==================================================
        sched_card, sched_box = self._make_card(
            "Physics schedule"
        )
        self._card_sched = sched_card
    
        self.lbl_sum_sched = QLabel("")
        self.lbl_sum_sched.setObjectName("sumLine")
    
        sched_body = QWidget(sched_card)
        sched_l = QVBoxLayout(sched_body)
        sched_l.setContentsMargins(0, 0, 0, 0)
        sched_l.setSpacing(6)
    
        self.sp_phys_warmup = QSpinBox()
        self.sp_phys_warmup.setRange(0, 10_000_000)
        _field(self.sp_phys_warmup)
    
        self.sp_phys_ramp = QSpinBox()
        self.sp_phys_ramp.setRange(0, 10_000_000)
        _field(self.sp_phys_ramp)
    
        msg = "Scale PDE residuals (stable gradients)"
        self.chk_scale_pde = QCheckBox(msg)
    
        grid_s = QGridLayout()
        grid_s.setHorizontalSpacing(10)
        grid_s.setVerticalSpacing(8)
    
        grid_s.addWidget(QLabel("Warmup steps:"), 0, 0)
        grid_s.addWidget(self.sp_phys_warmup, 0, 1)
        grid_s.addWidget(QLabel("Ramp steps:"), 1, 0)
        grid_s.addWidget(self.sp_phys_ramp, 1, 1)
        grid_s.addWidget(self.chk_scale_pde, 2, 0, 1, 2)
    
        grid_s.setColumnStretch(0, 0)
        grid_s.setColumnStretch(1, 1)
        sched_l.addLayout(grid_s)
    
        sched_hdr = QHBoxLayout()
        sched_hdr.setContentsMargins(0, 0, 0, 0)
        sched_hdr.setSpacing(8)
    
        sched_btn = self._mk_disclosure(
            summary=self.lbl_sum_sched,
            body=sched_body,
            expanded=False,
        )
        sched_hdr.addWidget(self.lbl_sum_sched, 1)
        sched_hdr.addWidget(sched_btn, 0)
    
        sched_box.addLayout(sched_hdr)
        sched_box.addWidget(sched_body)
    
        # ==================================================
        # Card: Advanced (dialog-only)
        # ==================================================
        self.chk_clean_stage1 = QCheckBox(
            "Clean Stage-1 run dir"
        )
        self.chk_clean_stage1.setVisible(False)
    
        adv_card, adv_box = self._make_card("Advanced")
        self._card_advopt = adv_card
    
        self.lbl_sum_advopt = QLabel("")
        self.lbl_sum_advopt.setObjectName("sumLine")
    
        adv_body = QWidget(adv_card)
        adv_l = QVBoxLayout(adv_body)
        adv_l.setContentsMargins(0, 0, 0, 0)
        adv_l.setSpacing(8)
    
        self.chk_eval_training = QCheckBox(
            "Evaluate training metrics"
        )
        self.chk_build_future = QCheckBox("Build future NPZ")
        self.chk_eval_training.setVisible(False)
        self.chk_build_future.setVisible(False)
    
        self.btn_train_options = QPushButton(
            "Advanced options..."
        )
        _btn(self.btn_train_options)
        adv_l.addWidget(self.btn_train_options, 0)
    
        hint = QLabel("Open dialog to edit options.")
        hint.setWordWrap(True)
        hint.setObjectName("hintLine")
        adv_l.addWidget(hint, 0)
    
        adv_hdr = QHBoxLayout()
        adv_hdr.setContentsMargins(0, 0, 0, 0)
        adv_hdr.setSpacing(8)
    
        adv_btn = self._mk_disclosure(
            summary=self.lbl_sum_advopt,
            body=adv_body,
            expanded=False,
        )
        adv_hdr.addWidget(self.lbl_sum_advopt, 1)
        adv_hdr.addWidget(adv_btn, 0)
    
        adv_box.addLayout(adv_hdr)
        adv_box.addWidget(adv_body)
    
        center_l.addWidget(temp_card)
        center_l.addWidget(train_card)
        center_l.addWidget(phys_card)
        center_l.addWidget(sched_card)
        center_l.addWidget(adv_card)
        center_l.addStretch(1)
    
        work_split.addWidget(center_scroll)
    
        # ----------------------------
        # Right: Run preview (scroll)
        # ----------------------------
        right_scroll = QScrollArea(right)
        right_scroll.setWidgetResizable(True)
        right_scroll.setFrameShape(QFrame.NoFrame)
    
        right_page = QWidget(right_scroll)
        right_scroll.setWidget(right_page)
    
        rp_l = QVBoxLayout(right_page)
        rp_l.setContentsMargins(0, 0, 0, 0)
        rp_l.setSpacing(10)
    
        preview_card, preview_box = self._make_card("Run preview")
        self.run_preview = RunPreviewPanel(
            store=self._store,
            show_compute=False,
        )
        preview_box.addWidget(self.run_preview, 1)
    
        rp_l.addWidget(preview_card)
        rp_l.addStretch(1)
        work_split.addWidget(right_scroll)
    
        # Sizes
        main_split.setStretchFactor(0, 0)
        main_split.setStretchFactor(1, 1)
        main_split.setSizes([260, 1200])
        main_split.setCollapsible(0, True)
    
        work_split.setStretchFactor(0, 1)
        work_split.setStretchFactor(1, 0)
        work_split.setSizes([860, 360])
        work_split.setCollapsible(1, True)
    
        main_split.addWidget(right)
    
        # ----------------------------
        # Bottom bar (full width)
        # ----------------------------
        self.lbl_run_status = QLabel("Ready.")
        self.lbl_run_status.setWordWrap(False)
        self.lbl_run_status.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Fixed,
        )
        self.lbl_run_status.setMinimumHeight(18)
        self.lbl_run_status.setMaximumHeight(18)
    
        self.train_btn = self._make_run_button("Run")
        self.lbl_run = QLabel("Run:")
        self.lbl_run.setAlignment(
            Qt.AlignRight | Qt.AlignVCenter
        )
    
        run_row = QHBoxLayout()
        run_row.setContentsMargins(0, 0, 0, 0)
        run_row.setSpacing(8)
        run_row.addStretch(1)
        run_row.addWidget(self.lbl_run)
        run_row.addWidget(self.train_btn)
    
        bottom = QHBoxLayout()
        bottom.setContentsMargins(0, 0, 0, 0)
        bottom.setSpacing(8)
        bottom.addWidget(self.lbl_run_status, 1)
        bottom.addLayout(run_row)
        outer.addLayout(bottom)
    
        # Nav wiring
        self.nav_list.currentRowChanged.connect(
            self._on_nav_changed
        )
        self.nav_list.setCurrentRow(0)
    
        self._refresh_summaries()

    def _fix_nav_list_height(self) -> None:
        if not hasattr(self, "nav_list"):
            return
        n = int(self.nav_list.count())
        if n <= 0:
            return
    
        row_h = int(self.nav_list.sizeHintForRow(0))
        if row_h <= 0:
            row_h = 34
    
        h = row_h * n
        h += 2 * int(self.nav_list.frameWidth())
        h += 8
        self.nav_list.setFixedHeight(h)

    def _build_presets(self) -> Dict[str, Dict[str, Any]]:
        return {
            # Do nothing preset (keeps current values)
            "Custom": {},

            "Balanced": {
                "epochs": 50,
                "batch_size": 32,
                "learning_rate": 1e-4,
                "pde_mode": "both",
                "lambda_cons": 0.10,
                "lambda_gw": 0.01,
                "lambda_prior": 0.10,
                "lambda_smooth": 0.01,
                "lambda_mv": 0.01,
                "physics_warmup_steps": 500,
                "physics_ramp_steps": 500,
                "scale_pde_residuals": True,
            },

            "Fast debug": {
                "epochs": 10,
                "batch_size": 64,
                "learning_rate": 5e-4,
                "pde_mode": "off",
                "lambda_cons": 0.0,
                "lambda_gw": 0.0,
                "lambda_prior": 0.0,
                "lambda_smooth": 0.0,
                "lambda_mv": 0.0,
                "physics_warmup_steps": 0,
                "physics_ramp_steps": 0,
                "scale_pde_residuals": False,
                "evaluate_training": False,
                "build_future_npz": False,
            },

            "Physics-heavy": {
                "epochs": 120,
                "batch_size": 32,
                "learning_rate": 1e-4,
                "pde_mode": "both",
                "lambda_cons": 1.0,
                "lambda_gw": 0.5,
                "lambda_prior": 0.2,
                "lambda_smooth": 0.1,
                "lambda_mv": 0.05,
                "physics_warmup_steps": 1000,
                "physics_ramp_steps": 2000,
                "scale_pde_residuals": True,
            },
        }

    def _apply_preset(self, name: str) -> None:
        preset = self._presets.get(name) or {}
        if not preset:
            return

        self._writing = True
        try:
            with self._store.batch():
                self._store.patch(dict(preset))
        finally:
            self._writing = False

        self.refresh_from_store()
        self.set_run_status(f"Preset applied: {name}")
        self._refresh_compute_nav()

    def _on_preset_changed(self, name: str) -> None:
        nm = str(name or "").strip() or "Custom"
        if nm == "Custom":
            return
        self._apply_preset(nm)

    # -----------------------------------------------------------------
    # Store helpers
    # -----------------------------------------------------------------
    def _get(self, key: str, default: Any) -> Any:
        try:
            return self._store.get_value(FieldKey(key))
        except Exception:
            return default

    def _set(self, key: str, value: Any) -> None:
        self._writing = True
        try:
            self._store.set_value_by_key(
                FieldKey(key),
                value,
            )
        finally:
            self._writing = False

    def _bind_spin(self, w: QSpinBox, key: str) -> None:
        def _push(v: int) -> None:
            self._set(key, int(v))

        w.valueChanged.connect(_push)

        def _pull() -> None:
            val = self._get(key, w.value())
            with QSignalBlocker(w):
                w.setValue(int(val))

        self._bindings[key] = _pull

    def _bind_dspin(self, w: QDoubleSpinBox, key: str) -> None:
        def _push(v: float) -> None:
            self._set(key, float(v))

        w.valueChanged.connect(_push)

        def _pull() -> None:
            val = self._get(key, w.value())
            with QSignalBlocker(w):
                w.setValue(float(val))

        self._bindings[key] = _pull

    def _bind_check(self, w: QCheckBox, key: str) -> None:
        def _push(v: bool) -> None:
            self._set(key, bool(v))

        w.toggled.connect(_push)

        def _pull() -> None:
            val = self._get(key, w.isChecked())
            with QSignalBlocker(w):
                w.setChecked(bool(val))

        self._bindings[key] = _pull

    def _bind_combo_text(self, w: QComboBox, key: str) -> None:
        def _push(_: int) -> None:
            self._set(key, str(w.currentText()))

        w.currentIndexChanged.connect(_push)

        def _pull() -> None:
            val = str(self._get(key, w.currentText()))
            idx = w.findText(val)
            if idx < 0:
                idx = 0
            with QSignalBlocker(w):
                w.setCurrentIndex(idx)

        self._bindings[key] = _pull

    # -----------------------------------------------------------------
    # Wiring
    # -----------------------------------------------------------------
    def _wire_ui(self) -> None:
        self._bind_spin(self.sp_train_end, "train_end_year")
        self._bind_spin(
            self.sp_forecast_start,
            "forecast_start_year",
        )
        self._bind_spin(
            self.sp_forecast_horizon,
            "forecast_horizon_years",
        )
        self._bind_spin(self.sp_time_steps, "time_steps")
    
        self._bind_spin(self.sp_epochs, "epochs")
        self._bind_spin(self.sp_batch_size, "batch_size")
        self._bind_dspin(self.sp_lr, "learning_rate")
    
        self._bind_combo_text(self.cmb_pde_mode, "pde_mode")
        self._bind_dspin(self.sb_lcons, "lambda_cons")
        self._bind_dspin(self.sb_lgw, "lambda_gw")
        self._bind_dspin(self.sb_lprior, "lambda_prior")
        self._bind_dspin(self.sb_lsmooth, "lambda_smooth")
        self._bind_dspin(self.sb_lmv, "lambda_mv")
    
        self._bind_spin(
            self.sp_phys_warmup,
            "physics_warmup_steps",
        )
        self._bind_spin(
            self.sp_phys_ramp,
            "physics_ramp_steps",
        )
        self._bind_check(
            self.chk_scale_pde,
            "scale_pde_residuals",
        )
    
        self._bind_check(
            self.chk_eval_training,
            "evaluate_training",
        )
        self._bind_check(
            self.chk_build_future,
            "build_future_npz",
        )
        self._bind_check(
            self.chk_clean_stage1,
            "clean_stage1_dir",
        )
    
        self.train_btn.clicked.connect(self.run_clicked)
        self.btn_train_options.clicked.connect(
            self.advanced_clicked
        )
        self.btn_features.clicked.connect(self.features_clicked)
        self.btn_arch.clicked.connect(self.arch_clicked)
        self.btn_prob.clicked.connect(self.prob_clicked)
        self.physics_btn.clicked.connect(self.physics_clicked)
    
        # Preset now lives in the HEAD
        self.head.preset_changed.connect(
            self._on_preset_changed
        )

    def _wire_store(self) -> None:
        # Store -> UI refresh
        def _on_store_changed(*_: Any) -> None:
            if self._writing:
                return
            self.refresh_from_store()

        # Some stores expose only config_changed, others also
        # expose config_replaced. Hook both if available.
        try:
            self._store.config_changed.connect(_on_store_changed)
        except Exception:
            pass

        try:
            self._store.config_replaced.connect(_on_store_changed)
        except Exception:
            pass

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------
    def refresh_from_store(self) -> None:
        for pull in self._bindings.values():
            try:
                pull()
            except Exception:
                continue
    
        self._refresh_summaries()
    
        if hasattr(self, "run_preview"):
            self._refresh_compute_nav()


    def set_run_status(self, text: str) -> None:
        if not hasattr(self, "lbl_run_status"):
            return
        fm = QFontMetrics(self.lbl_run_status.font())
        el = fm.elidedText(
            str(text),
            Qt.ElideRight,
            self.lbl_run_status.width(),
        )
        self.lbl_run_status.setText(el)
        
    def _refresh_compute_nav(self) -> None:
        if not hasattr(self, "lbl_compute_nav"):
            return
        self.lbl_compute_nav.setText(
            runtime_summary_text(self._store)
        )
    
        

