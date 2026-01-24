# geoprior/ui/inference/center_panel.py
# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>

from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple

from PyQt5.QtCore import Qt, QSignalBlocker, pyqtSignal
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
    QStyle, 
    QSizePolicy
)

from ...config.prior_schema import CHOICE_SPECS, FieldKey
from ...config.store import GeoConfigStore
from ..icon_utils import try_icon

MakeCardFn = Callable[[str], Tuple[QWidget, QVBoxLayout]]

__all__ = ["InferenceCenterPanel"]


class InferenceCenterPanel(QWidget):
    """
    Center [C]: editable cards, store-backed.

    - Store-backed fields write immediately to GeoConfigStore.
    - Runtime-only paths stay in widgets (controller reads).
    """

    runtime_changed = pyqtSignal()

    browse_model_clicked = pyqtSignal()
    browse_manifest_clicked = pyqtSignal()
    browse_inputs_clicked = pyqtSignal()
    browse_targets_clicked = pyqtSignal()
    browse_calib_clicked = pyqtSignal()

    def __init__(
        self,
        *,
        store: GeoConfigStore,
        make_card: MakeCardFn,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)

        self._store = store
        self._make_card = make_card
        self._cards: Dict[str, QWidget] = {}

        self._build_ui()
        self._wire()
        self.refresh_from_store()

    # ---------------------------------------------------------
    # Public
    # ---------------------------------------------------------
    def refresh_from_store(self) -> None:
        s = self._store

        with QSignalBlocker(self.sp_inf_cov):
            v = s.get_value(FieldKey("interval_level"), default=0.80)
            try:
                self.sp_inf_cov.setValue(float(v))
            except Exception:
                pass

        with QSignalBlocker(self.cmb_calib_mode):
            v = s.get_value(
                FieldKey("calibration_mode"),
                default="none",
            )
            idx = self.cmb_calib_mode.findData(str(v))
            if idx >= 0:
                self.cmb_calib_mode.setCurrentIndex(idx)

        with QSignalBlocker(self.sp_calib_temp):
            v = s.get_value(
                FieldKey("calibration_temperature"),
                default=1.0,
            )
            try:
                self.sp_calib_temp.setValue(float(v))
            except Exception:
                pass

        with QSignalBlocker(self.sp_cross_pen):
            v = s.get_value(
                FieldKey("crossing_penalty"),
                default=0.0,
            )
            try:
                self.sp_cross_pen.setValue(float(v))
            except Exception:
                pass

        with QSignalBlocker(self.sp_cross_margin):
            v = s.get_value(
                FieldKey("crossing_margin"),
                default=0.0,
            )
            try:
                self.sp_cross_margin.setValue(float(v))
            except Exception:
                pass
        
        self._refresh_summaries()

    def focus_section(self, key: str) -> None:
        w = self._cards.get(key)
        if w is None:
            return
        w.setFocus()

        # If we are inside a scroll area, ensure visible.
        if self._scroll is not None:
            self._scroll.ensureWidgetVisible(w)
            
    def _refresh_summaries(self) -> None:
        """
        Update the per-card summary lines from store + runtime widgets.
    
        This mirrors Train/Tune: short, informative, and resilient
        (never raises if some widgets are missing).
        """
        # -------------------------
        # Artifacts (runtime)
        # -------------------------
        try:
            ds = str(self.cmb_inf_dataset.currentData() or "—")
            fut = bool(self.chk_inf_use_future.isChecked())
            bsz = int(self.sp_inf_batch.value())
            model_ok = bool(self.inf_model_edit.text().strip())
            mani = self.inf_manifest_edit.text().strip()
    
            model_s = "model✓" if model_ok else "model—"
            mani_s = "manifest:auto" if not mani else "manifest:✓"
            ds_s = f"ds={ds}{'+future' if fut else ''}"
            self.lbl_sum_art.setText(
                f"{model_s}  •  {mani_s}  •  {ds_s}  •  batch={bsz}"
            )
        except Exception:
            pass
    
        # -------------------------
        # Uncertainty (store-backed)
        # -------------------------
        try:
            cov = float(self.sp_inf_cov.value())
            mode = str(self.cmb_calib_mode.currentData() or "none")
            tmp = float(self.sp_calib_temp.value())
            pen = float(self.sp_cross_pen.value())
            mar = float(self.sp_cross_margin.value())
    
            # Keep it compact; crossing only if non-zero-ish.
            cross = ""
            if abs(pen) > 1e-12 or abs(mar) > 1e-12:
                cross = f"  •  cross(p={pen:.2g}, m={mar:.2g})"
    
            self.lbl_sum_unc.setText(
                f"cov={cov:.2f}  •  mode={mode}  •  T={tmp:.2g}{cross}"
            )
        except Exception:
            pass

        # -------------------------
        # Calibration (runtime)
        # -------------------------
        try:
            src = bool(self.chk_inf_use_source_calib.isChecked())
            fit = bool(self.chk_inf_fit_calib.isChecked())
            cal = self.inf_calib_edit.text().strip()
    
            src_s = "source:ON" if src else "source:OFF"
            fit_s = "fit:ON" if fit else "fit:OFF"
            file_s = "file:✓" if cal else "file:none"
    
            self.lbl_sum_cal.setText(f"{src_s}  •  {fit_s}  •  {file_s}")
        except Exception:
            pass
    
        # -------------------------
        # Outputs (runtime)
        # -------------------------
        try:
            gwl = bool(self.chk_inf_include_gwl.isChecked())
            pl = bool(self.chk_inf_plots.isChecked())
    
            gwl_s = "GWL:ON" if gwl else "GWL:OFF"
            pl_s = "plots:ON" if pl else "plots:OFF"
    
            self.lbl_sum_out.setText(f"{gwl_s}  •  {pl_s}")
        except Exception:
            pass
    
        # -------------------------
        # Advanced (static / placeholder)
        # -------------------------
        try:
            # Until you add real knobs, keep it consistent.
            self.lbl_sum_adv.setText("extra toggles, future knobs")
        except Exception:
            pass

    # ---------------------------------------------------------
    # Store helpers
    # ---------------------------------------------------------
    def _store_set(self, key: str, value) -> None:
        try:
            self._store.set_value_by_key(
                FieldKey(key),
                value,
            )
        except Exception:
            return

    # ---------------------------------------------------------
    # UI
    # ---------------------------------------------------------
    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
    
        self._scroll = QScrollArea(self)
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QFrame.NoFrame)
    
        page = QWidget(self._scroll)
        self._scroll.setWidget(page)
    
        lay = QVBoxLayout(page)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(10)
    
        # ---------------------------------------------------------
        # Small helper: browse toolbutton with icon + fallback
        # ---------------------------------------------------------
        def _mk_browse_btn(
            tip: str,
            icon_names: Tuple[str, ...],
            fallback: QStyle.StandardPixmap,
        ) -> QToolButton:
            b = QToolButton(self)
            b.setObjectName("miniAction")
            b.setAutoRaise(True)
            b.setCursor(Qt.PointingHandCursor)
            b.setToolButtonStyle(Qt.ToolButtonIconOnly)
            b.setToolTip(tip)
            b.setFixedSize(30, 26)
    
            ic = None
            for nm in icon_names:
                ic = try_icon(nm)
                if ic is not None:
                    break
            if ic is None:
                ic = self.style().standardIcon(fallback)
            b.setIcon(ic)
            b.setIconSize(b.iconSize())
            return b
    
        # =========================================================
        # Card 1: Artifacts
        # =========================================================
        c1, b1 = self._make_card("Artifacts")
        self._cards["artifacts"] = c1
    
        # Header: summary + disclosure
        sum_row = QWidget(c1)
        sum_l = QHBoxLayout(sum_row)
        sum_l.setContentsMargins(0, 0, 0, 0)
        sum_l.setSpacing(8)
    
        self.lbl_sum_art = QLabel(
            "model, manifest, dataset/custom inputs",
            sum_row,
        )
        self.lbl_sum_art.setObjectName("sumLine")
        self.lbl_sum_art.setWordWrap(True)
    
        self._art_body = QWidget(c1)
        self._art_body.setObjectName("drawer")
    
        btn_art = self._mk_disclosure(
            summary=self.lbl_sum_art,
            body=self._art_body,
            expanded=True,  # artifacts is essential -> open by default
        )
    
        sum_l.addWidget(self.lbl_sum_art, 1)
        sum_l.addWidget(btn_art, 0)
        b1.addWidget(sum_row, 0)
    
        art_l = QVBoxLayout(self._art_body)
        art_l.setContentsMargins(0, 4, 0, 0)
        art_l.setSpacing(8)
    
        self.inf_model_edit = QLineEdit(self._art_body)
        self.inf_model_edit.setPlaceholderText("Select .keras model…")
        self.inf_model_btn = _mk_browse_btn(
            "Browse model (.keras)",
            ("input.svg", "browse.svg", "folder_open.svg"),
            QStyle.SP_DirOpenIcon,
        )
    
        self.inf_manifest_edit = QLineEdit(self._art_body)
        self.inf_manifest_edit.setPlaceholderText(
            "Stage-1 manifest (auto if empty)"
        )
        self.inf_manifest_btn = _mk_browse_btn(
            "Browse Stage-1 manifest",
            ("browse.svg", "folder_open.svg", "open.svg"),
            QStyle.SP_DirOpenIcon,
        )
    
        self.cmb_inf_dataset = QComboBox(self._art_body)
        self.cmb_inf_dataset.addItem("Validation (val)", "val")
        self.cmb_inf_dataset.addItem("Test (test)", "test")
        self.cmb_inf_dataset.addItem("Train (train)", "train")
        self.cmb_inf_dataset.addItem("Custom NPZ", "custom")
    
        self.chk_inf_use_future = QCheckBox(
            "Use Stage-1 future NPZ (forecast mode)",
            self._art_body,
        )
    
        self.inf_inputs_edit = QLineEdit(self._art_body)
        self.inf_inputs_edit.setPlaceholderText("Custom inputs .npz")
        self.inf_inputs_btn = _mk_browse_btn(
            "Browse inputs NPZ",
            ("input.svg", "browse.svg", "open.svg"),
            QStyle.SP_DialogOpenButton,
        )
    
        self.inf_targets_edit = QLineEdit(self._art_body)
        self.inf_targets_edit.setPlaceholderText(
            "Optional targets .npz (for metrics)"
        )
        self.inf_targets_btn = _mk_browse_btn(
            "Browse targets NPZ",
            ("target.svg", "browse.svg", "open.svg"),
            QStyle.SP_DialogOpenButton,
        )
    
        self.sp_inf_batch = QSpinBox(self._art_body)
        self.sp_inf_batch.setRange(1, 2048)
        self.sp_inf_batch.setValue(32)
    
        g1 = QGridLayout()
        g1.setContentsMargins(0, 0, 0, 0)
        g1.setHorizontalSpacing(10)
        g1.setVerticalSpacing(8)
        g1.setColumnStretch(0, 0)
        g1.setColumnStretch(1, 1)
        g1.setColumnStretch(2, 0)
    
        r = 0
        g1.addWidget(QLabel("Model file:"), r, 0)
        g1.addWidget(self.inf_model_edit, r, 1)
        g1.addWidget(self.inf_model_btn, r, 2)
        r += 1
    
        g1.addWidget(QLabel("Stage-1 manifest:"), r, 0)
        g1.addWidget(self.inf_manifest_edit, r, 1)
        g1.addWidget(self.inf_manifest_btn, r, 2)
        r += 1
    
        g1.addWidget(QLabel("Dataset:"), r, 0)
        g1.addWidget(self.cmb_inf_dataset, r, 1, 1, 2)
        r += 1
    
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(10)
        row.addWidget(self.chk_inf_use_future)
        row.addStretch(1)
        row.addWidget(QLabel("Batch size:"))
        row.addWidget(self.sp_inf_batch)
    
        g1.addLayout(row, r, 0, 1, 3)
        r += 1
    
        g1.addWidget(QLabel("Custom inputs:"), r, 0)
        g1.addWidget(self.inf_inputs_edit, r, 1)
        g1.addWidget(self.inf_inputs_btn, r, 2)
        r += 1
    
        g1.addWidget(QLabel("Custom targets:"), r, 0)
        g1.addWidget(self.inf_targets_edit, r, 1)
        g1.addWidget(self.inf_targets_btn, r, 2)
    
        art_l.addLayout(g1)
        b1.addWidget(self._art_body, 0)
        lay.addWidget(c1, 0)
    
        # =========================================================
        # Card 2: Uncertainty
        # =========================================================
        c2, b2 = self._make_card("Uncertainty")
        self._cards["uncertainty"] = c2
    
        sum_row = QWidget(c2)
        sum_l = QHBoxLayout(sum_row)
        sum_l.setContentsMargins(0, 0, 0, 0)
        sum_l.setSpacing(8)
    
        self.lbl_sum_unc = QLabel(
            "interval, mode, temperature, crossing",
            sum_row,
        )
        self.lbl_sum_unc.setObjectName("sumLine")
        self.lbl_sum_unc.setWordWrap(True)
    
        self._unc_body = QWidget(c2)
        self._unc_body.setObjectName("drawer")
    
        btn_unc = self._mk_disclosure(
            summary=self.lbl_sum_unc,
            body=self._unc_body,
            expanded=False,
        )
    
        sum_l.addWidget(self.lbl_sum_unc, 1)
        sum_l.addWidget(btn_unc, 0)
        b2.addWidget(sum_row, 0)
    
        unc_l = QVBoxLayout(self._unc_body)
        unc_l.setContentsMargins(0, 4, 0, 0)
        unc_l.setSpacing(8)
    
        self.sp_inf_cov = QDoubleSpinBox(self._unc_body)
        self.sp_inf_cov.setDecimals(3)
        self.sp_inf_cov.setRange(0.50, 0.99)
        self.sp_inf_cov.setSingleStep(0.01)
        self.sp_inf_cov.setValue(0.80)
    
        self.cmb_calib_mode = QComboBox(self._unc_body)
        modes = CHOICE_SPECS.get("calibration_mode", ("none",))
        for m in modes:
            self.cmb_calib_mode.addItem(m, m)
    
        self.sp_calib_temp = QDoubleSpinBox(self._unc_body)
        self.sp_calib_temp.setDecimals(3)
        self.sp_calib_temp.setRange(0.01, 100.0)
        self.sp_calib_temp.setSingleStep(0.10)
        self.sp_calib_temp.setValue(1.0)
    
        self.sp_cross_pen = QDoubleSpinBox(self._unc_body)
        self.sp_cross_pen.setDecimals(4)
        self.sp_cross_pen.setRange(0.0, 1e6)
        self.sp_cross_pen.setSingleStep(0.1)
        self.sp_cross_pen.setValue(0.0)
    
        self.sp_cross_margin = QDoubleSpinBox(self._unc_body)
        self.sp_cross_margin.setDecimals(4)
        self.sp_cross_margin.setRange(0.0, 1e6)
        self.sp_cross_margin.setSingleStep(0.1)
        self.sp_cross_margin.setValue(0.0)
    
        g2 = QGridLayout()
        g2.setContentsMargins(0, 0, 0, 0)
        g2.setHorizontalSpacing(10)
        g2.setVerticalSpacing(8)
        g2.setColumnStretch(0, 0)
        g2.setColumnStretch(1, 1)
    
        rr = 0
        g2.addWidget(QLabel("Interval level:"), rr, 0)
        g2.addWidget(self.sp_inf_cov, rr, 1)
        rr += 1
    
        g2.addWidget(QLabel("Calibration mode:"), rr, 0)
        g2.addWidget(self.cmb_calib_mode, rr, 1)
        rr += 1
    
        g2.addWidget(QLabel("Temperature:"), rr, 0)
        g2.addWidget(self.sp_calib_temp, rr, 1)
        rr += 1
    
        g2.addWidget(QLabel("Cross penalty:"), rr, 0)
        g2.addWidget(self.sp_cross_pen, rr, 1)
        rr += 1
    
        g2.addWidget(QLabel("Cross margin:"), rr, 0)
        g2.addWidget(self.sp_cross_margin, rr, 1)
    
        unc_l.addLayout(g2)
        b2.addWidget(self._unc_body, 0)
        lay.addWidget(c2, 0)
    
        # =========================================================
        # Card 3: Calibration
        # =========================================================
        c3, b3 = self._make_card("Calibration")
        self._cards["calibration"] = c3
    
        sum_row = QWidget(c3)
        sum_l = QHBoxLayout(sum_row)
        sum_l.setContentsMargins(0, 0, 0, 0)
        sum_l.setSpacing(8)
    
        self.lbl_sum_cal = QLabel(
            "source calibrator, fit-on-val, file",
            sum_row,
        )
        self.lbl_sum_cal.setObjectName("sumLine")
        self.lbl_sum_cal.setWordWrap(True)
    
        self._cal_body = QWidget(c3)
        self._cal_body.setObjectName("drawer")
    
        btn_cal = self._mk_disclosure(
            summary=self.lbl_sum_cal,
            body=self._cal_body,
            expanded=False,
        )
    
        sum_l.addWidget(self.lbl_sum_cal, 1)
        sum_l.addWidget(btn_cal, 0)
        b3.addWidget(sum_row, 0)
    
        cal_l = QVBoxLayout(self._cal_body)
        cal_l.setContentsMargins(0, 4, 0, 0)
        cal_l.setSpacing(8)
    
        self.chk_inf_use_source_calib = QCheckBox(
            "Use source calibrator (interval_factors_80.npy)",
            self._cal_body,
        )
        self.chk_inf_fit_calib = QCheckBox(
            "Fit calibrator on validation split",
            self._cal_body,
        )
    
        self.inf_calib_edit = QLineEdit(self._cal_body)
        self.inf_calib_edit.setPlaceholderText(
            "Optional explicit calibrator .npy"
        )
        self.inf_calib_btn = _mk_browse_btn(
            "Browse calibrator file",
            ("browse.svg", "target.svg", "open.svg"),
            QStyle.SP_DialogOpenButton,
        )
    
        g3 = QGridLayout()
        g3.setContentsMargins(0, 0, 0, 0)
        g3.setHorizontalSpacing(10)
        g3.setVerticalSpacing(8)
        g3.setColumnStretch(0, 0)
        g3.setColumnStretch(1, 1)
        g3.setColumnStretch(2, 0)
    
        r3 = 0
        g3.addWidget(self.chk_inf_use_source_calib, r3, 0, 1, 3)
        r3 += 1
        g3.addWidget(self.chk_inf_fit_calib, r3, 0, 1, 3)
        r3 += 1
    
        g3.addWidget(QLabel("Calibrator file:"), r3, 0)
        g3.addWidget(self.inf_calib_edit, r3, 1)
        g3.addWidget(self.inf_calib_btn, r3, 2)
    
        cal_l.addLayout(g3)
        b3.addWidget(self._cal_body, 0)
        lay.addWidget(c3, 0)
    
        # =========================================================
        # Card 4: Outputs
        # =========================================================
        c4, b4 = self._make_card("Outputs")
        self._cards["outputs"] = c4
    
        sum_row = QWidget(c4)
        sum_l = QHBoxLayout(sum_row)
        sum_l.setContentsMargins(0, 0, 0, 0)
        sum_l.setSpacing(8)
    
        self.lbl_sum_out = QLabel("include GWL, plots", sum_row)
        self.lbl_sum_out.setObjectName("sumLine")
        self.lbl_sum_out.setWordWrap(True)
    
        self._out_body = QWidget(c4)
        self._out_body.setObjectName("drawer")
    
        btn_out = self._mk_disclosure(
            summary=self.lbl_sum_out,
            body=self._out_body,
            expanded=False,
        )
    
        sum_l.addWidget(self.lbl_sum_out, 1)
        sum_l.addWidget(btn_out, 0)
        b4.addWidget(sum_row, 0)
    
        out_l = QVBoxLayout(self._out_body)
        out_l.setContentsMargins(0, 4, 0, 0)
        out_l.setSpacing(8)
    
        self.chk_inf_include_gwl = QCheckBox(
            "Include GWL columns in CSV",
            self._out_body,
        )
        self.chk_inf_plots = QCheckBox(
            "Generate plots",
            self._out_body,
        )
        self.chk_inf_plots.setChecked(True)
    
        out_l.addWidget(self.chk_inf_include_gwl, 0)
        out_l.addWidget(self.chk_inf_plots, 0)
        out_l.addStretch(1)
    
        b4.addWidget(self._out_body, 0)
        lay.addWidget(c4, 0)
    
        # =========================================================
        # Card 5: Advanced
        # =========================================================
        c5, b5 = self._make_card("Advanced")
        self._cards["advanced"] = c5
    
        sum_row = QWidget(c5)
        sum_l = QHBoxLayout(sum_row)
        sum_l.setContentsMargins(0, 0, 0, 0)
        sum_l.setSpacing(8)
    
        self.lbl_sum_adv = QLabel("extra toggles, future knobs", sum_row)
        self.lbl_sum_adv.setObjectName("sumLine")
        self.lbl_sum_adv.setWordWrap(True)
    
        self._adv_body = QWidget(c5)
        self._adv_body.setObjectName("drawer")
    
        btn_adv = self._mk_disclosure(
            summary=self.lbl_sum_adv,
            body=self._adv_body,
            expanded=False,
        )
    
        sum_l.addWidget(self.lbl_sum_adv, 1)
        sum_l.addWidget(btn_adv, 0)
        b5.addWidget(sum_row, 0)
    
        adv_l = QVBoxLayout(self._adv_body)
        adv_l.setContentsMargins(0, 4, 0, 0)
        adv_l.setSpacing(8)
    
        self.btn_inf_options = QToolButton(self._adv_body)
        self.btn_inf_options.setObjectName("miniAction")
        self.btn_inf_options.setAutoRaise(True)
        self.btn_inf_options.setCursor(Qt.PointingHandCursor)
        self.btn_inf_options.setToolButtonStyle(Qt.ToolButtonTextOnly)
        self.btn_inf_options.setText("Advanced options…")
    
        adv_l.addWidget(self.btn_inf_options, 0)
        adv_l.addStretch(1)
    
        b5.addWidget(self._adv_body, 0)
        lay.addWidget(c5, 0)
    
        lay.addStretch(1)
        root.addWidget(self._scroll, 1)
    
        self._update_widgets_state()
        
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
            QSizePolicy.Minimum,
        )
        return btn

    def _add_card_header(
        self,
        body: QVBoxLayout,
        summary: str,
        *,
        key: str,
        body_widget: Optional[QWidget] = None,
    ) -> QToolButton:
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(8)
    
        sm = QLabel(summary, self)
        sm.setObjectName("sumLine")
    
        btn = QToolButton(self)
        btn.setObjectName("disclosure")
        btn.setText("Edit")
        btn.setCheckable(True)
        btn.setChecked(True)
    
        row.addWidget(sm, 1)
        row.addWidget(btn, 0)
    
        body.addLayout(row)
    
        # generic disclosure wiring (no dependence on attributes)
        if body_widget is not None:
            btn.toggled.connect(body_widget.setVisible)
    
        return btn


    # ---------------------------------------------------------
    # Wiring
    # ---------------------------------------------------------
    def _wire(self) -> None:
        """
        Centralized wiring with no duplication.
    
        Conventions
        -----------
        - Runtime widgets -> emit runtime_changed + refresh summaries.
        - Dataset/future toggles -> update enabled state, then refresh.
        - Store-backed widgets -> write to store via _on_store_field_changed.
        """
    
        # --------------------------
        # Browse clicks (signals out)
        # --------------------------
        self.inf_model_btn.clicked.connect(
            self.browse_model_clicked.emit
        )
        self.inf_manifest_btn.clicked.connect(
            self.browse_manifest_clicked.emit
        )
        self.inf_inputs_btn.clicked.connect(
            self.browse_inputs_clicked.emit
        )
        self.inf_targets_btn.clicked.connect(
            self.browse_targets_clicked.emit
        )
        self.inf_calib_btn.clicked.connect(
            self.browse_calib_clicked.emit
        )
    
        # --------------------------
        # Runtime changes
        # --------------------------
        def _runtime_bump(*_a) -> None:
            self.runtime_changed.emit()
            self._refresh_summaries()
    
        for w in (
            self.inf_model_edit,
            self.inf_manifest_edit,
            self.inf_inputs_edit,
            self.inf_targets_edit,
            self.inf_calib_edit,
        ):
            w.textChanged.connect(_runtime_bump)
    
        for w in (
            self.chk_inf_use_source_calib,
            self.chk_inf_fit_calib,
            self.chk_inf_include_gwl,
            self.chk_inf_plots,
        ):
            w.toggled.connect(_runtime_bump)
    
        self.sp_inf_batch.valueChanged.connect(_runtime_bump)
    
        # Dataset selection affects which runtime fields are enabled
        self.cmb_inf_dataset.currentIndexChanged.connect(
            self._update_widgets_state
        )
        self.chk_inf_use_future.toggled.connect(
            self._update_widgets_state
        )
    
        # --------------------------
        # Store-backed changes
        # --------------------------
        self.sp_inf_cov.valueChanged.connect(
            lambda v: self._on_store_field_changed(
                "interval_level",
                float(v),
            )
        )
    
        self.cmb_calib_mode.currentIndexChanged.connect(
            lambda _i: self._on_store_field_changed(
                "calibration_mode",
                str(self.cmb_calib_mode.currentData() or "none"),
            )
        )
    
        self.sp_calib_temp.valueChanged.connect(
            lambda v: self._on_store_field_changed(
                "calibration_temperature",
                float(v),
            )
        )
    
        self.sp_cross_pen.valueChanged.connect(
            lambda v: self._on_store_field_changed(
                "crossing_penalty",
                float(v),
            )
        )
    
        self.sp_cross_margin.valueChanged.connect(
            lambda v: self._on_store_field_changed(
                "crossing_margin",
                float(v),
            )
        )
    
        # Initial summaries
        self._refresh_summaries()


    def _on_store_field_changed(self, key: str, value) -> None:
        self._store_set(key, value)
        self._refresh_summaries()
        
    def _update_widgets_state(self) -> None:
        key = self.cmb_inf_dataset.currentData() or "test"
        use_future = self.chk_inf_use_future.isChecked()

        custom = (key == "custom") and (not use_future)

        self.inf_inputs_edit.setEnabled(custom)
        self.inf_inputs_btn.setEnabled(custom)

        self.inf_targets_edit.setEnabled(custom)
        self.inf_targets_btn.setEnabled(custom)

        self.runtime_changed.emit()
        self._refresh_summaries()
