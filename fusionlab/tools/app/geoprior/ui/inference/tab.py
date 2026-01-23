# geoprior/ui/inference/tab.py
# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>

from __future__ import annotations

import os
from typing import Any, Callable, Dict, Optional, Tuple

from PyQt5.QtCore import (
    Qt,
    QSignalBlocker,
    QUrl,
    pyqtSignal,
)
from PyQt5.QtGui import QDesktopServices, QFont
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QToolButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ...config.store import GeoConfigStore
from ...config.prior_schema import CHOICE_SPECS, FieldKey
from ...config.store import GeoConfigStore

from .plan import build_plan_text
from .status import compute_infer_nav  
 
# from .head import InferenceHeadBar
# from .navigator import InferenceNavigator
# from .center_panel import InferenceCenterPanel
# from .preview import InferencePreviewPanel
# from .details import InferenceDetailsPanel
# from .status import InferenceStatusBar

MakeCardFn = Callable[[str], Tuple[QWidget, QVBoxLayout]]
MakeRunBtnFn = Callable[[str], QWidget]

__all__ = ["InferenceTab"]


def _exists(p: str) -> bool:
    try:
        return bool(p) and os.path.exists(p)
    except Exception:
        return False


def _abspath(p: str) -> str:
    try:
        return os.path.abspath(p)
    except Exception:
        return p


__all__ = ["InferenceTab"]


class InferenceTab(QWidget):
    """
    Store-aware Inference tab (Train-like layout).

    Backward compatibility:
    - preserves legacy widget attributes
    - preserves signature with make_card/make_run_button
    """

    run_clicked = pyqtSignal()
    advanced_clicked = pyqtSignal()

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
        make_run_button: MakeRunBtnFn,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)

        self._store = store
        self._make_card = make_card
        self._make_run_button = make_run_button

        self._last_outputs: Dict[str, str] = {}

        self._build_ui()
        self._connect_ui()
        self._bind_store()

        self.refresh_from_store()
        self._update_widgets_state()
        self._update_preview()
        self._refresh_nav_chips()

    # ----------------------------------------------------------
    # Public helpers
    # ----------------------------------------------------------
    def refresh_from_store(self) -> None:
        self._sync_ui_from_store()
        self._update_preview()
        self._refresh_nav_chips()

    def set_last_outputs(self, outputs: Dict[str, str]) -> None:
        out = dict(outputs or {})

        if "forecast_csv" in out and "csv_future_path" not in out:
            out["csv_future_path"] = out.get("forecast_csv", "")

        self._last_outputs = out
        self._sync_last_outputs_ui()
        self._update_preview()

    # ----------------------------------------------------------
    # UI
    # ----------------------------------------------------------
    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(8)

        # Main split: left vs work
        main = QHBoxLayout()
        main.setSpacing(10)

        left = self._build_left_col()
        work = self._build_work_col()

        main.addWidget(left, 0)
        main.addWidget(work, 1)

        root.addLayout(main, 1)

        # Bottom status row (no Run here)
        self._lbl_status = QLabel("Ready.", self)
        self._lbl_status.setObjectName("sumLine")

        bot = QHBoxLayout()
        bot.setContentsMargins(0, 0, 0, 0)
        bot.addWidget(self._lbl_status, 1)

        root.addLayout(bot, 0)

        # Back-compat run button attribute:
        # keep it hidden (global Run in app.py).
        self.btn_run_infer = self._make_run_button("Run inference")
        self.btn_run_infer.setVisible(False)

    def _build_left_col(self) -> QWidget:
        box = QWidget(self)
        lay = QVBoxLayout(box)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(8)

        # [A] Navigator card
        nav_card = QFrame(self)
        nav_card.setObjectName("trainNavCard")
        nav_card.setFrameShape(QFrame.StyledPanel)

        nav_l = QVBoxLayout(nav_card)
        nav_l.setContentsMargins(10, 10, 10, 10)
        nav_l.setSpacing(8)

        nav_t = QLabel("Setup checklist", self)
        nav_t.setObjectName("trainNavTitle")
        nav_l.addWidget(nav_t, 0)

        self.nav_list = QListWidget(self)
        self.nav_list.setObjectName("trainNavList")
        self.nav_list.setSelectionMode(
            self.nav_list.SingleSelection
        )
        nav_l.addWidget(self.nav_list, 1)

        lay.addWidget(nav_card, 1)

        # [E] Extras card (last outputs + shortcuts)
        extra, extra_body = self._make_card("Artifacts & options")
        lay.addWidget(extra, 0)

        self.ed_last_dir = QLineEdit(self)
        self.ed_last_dir.setReadOnly(True)
        self.btn_open_last_dir = QPushButton("Open folder", self)
        self.btn_open_last_dir.setEnabled(False)

        self.ed_last_eval = QLineEdit(self)
        self.ed_last_eval.setReadOnly(True)
        self.btn_open_last_eval = QPushButton("Open CSV", self)
        self.btn_open_last_eval.setEnabled(False)

        self.ed_last_future = QLineEdit(self)
        self.ed_last_future.setReadOnly(True)
        self.btn_open_last_future = QPushButton("Open CSV", self)
        self.btn_open_last_future.setEnabled(False)

        self.ed_last_json = QLineEdit(self)
        self.ed_last_json.setReadOnly(True)
        self.btn_open_last_json = QPushButton("Open JSON", self)
        self.btn_open_last_json.setEnabled(False)

        g = QGridLayout()
        rr = 0

        g.addWidget(QLabel("Last output dir:"), rr, 0)
        g.addWidget(self.ed_last_dir, rr, 1)
        g.addWidget(self.btn_open_last_dir, rr, 2)
        rr += 1

        g.addWidget(QLabel("Eval CSV:"), rr, 0)
        g.addWidget(self.ed_last_eval, rr, 1)
        g.addWidget(self.btn_open_last_eval, rr, 2)
        rr += 1

        g.addWidget(QLabel("Future CSV:"), rr, 0)
        g.addWidget(self.ed_last_future, rr, 1)
        g.addWidget(self.btn_open_last_future, rr, 2)
        rr += 1

        g.addWidget(QLabel("Summary JSON:"), rr, 0)
        g.addWidget(self.ed_last_json, rr, 1)
        g.addWidget(self.btn_open_last_json, rr, 2)

        extra_body.addLayout(g)
        extra_body.addStretch(1)

        # Back-compat options button (visible)
        self.btn_inf_options = QToolButton(self)
        self.btn_inf_options.setObjectName("miniAction")
        self.btn_inf_options.setText("Advanced options…")
        self.btn_inf_options.setToolButtonStyle(
            Qt.ToolButtonTextOnly
        )
        self.btn_inf_options.setAutoRaise(True)

        extra_body.addWidget(self.btn_inf_options, 0)

        return box

    def _build_work_col(self) -> QWidget:
        box = QWidget(self)
        lay = QVBoxLayout(box)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(8)

        # [B] Head bar
        top = QWidget(self)
        top.setObjectName("trainTopBar")
        top_l = QHBoxLayout(top)
        top_l.setContentsMargins(8, 6, 8, 6)
        top_l.setSpacing(8)

        self._plan_sum = QLabel("", self)
        self._plan_sum.setObjectName("sumLine")
        self._plan_sum.setWordWrap(False)

        self._btn_refresh = QToolButton(self)
        self._btn_refresh.setObjectName("miniAction")
        self._btn_refresh.setText("Refresh")
        self._btn_refresh.setAutoRaise(True)

        top_l.addWidget(self._plan_sum, 1)
        top_l.addWidget(self._btn_refresh, 0)

        lay.addWidget(top, 0)

        # [C] Center cards + [D] Preview
        mid = QHBoxLayout()
        mid.setSpacing(10)

        center = self._build_center_cards()
        prev = self._build_preview()

        mid.addWidget(center, 1)
        mid.addWidget(prev, 1)

        lay.addLayout(mid, 1)

        # Navigator items (must exist after center/prev)
        self._init_nav_items()

        return box

    def _build_center_cards(self) -> QWidget:
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)

        page = QWidget(scroll)
        scroll.setWidget(page)

        root = QVBoxLayout(page)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(10)
        root.addStretch(0)

        # Card: Artifacts
        c1, b1 = self._make_card("Artifacts")

        sum1 = QLabel(
            "model, manifest, dataset/custom inputs",
            self,
        )
        sum1.setObjectName("sumLine")

        self._disc_art = QToolButton(self)
        self._disc_art.setObjectName("disclosure")
        self._disc_art.setText("Edit")
        self._disc_art.setCheckable(True)
        self._disc_art.setChecked(True)

        h1 = QHBoxLayout()
        h1.addWidget(sum1, 1)
        h1.addWidget(self._disc_art, 0)
        b1.addLayout(h1)

        self.inf_model_edit = QLineEdit(self)
        self.inf_model_edit.setPlaceholderText(
            "Select .keras model..."
        )
        self.inf_model_btn = QPushButton("Browse...", self)

        self.inf_manifest_edit = QLineEdit(self)
        self.inf_manifest_edit.setPlaceholderText(
            "Stage-1 manifest (auto if empty)"
        )
        self.inf_manifest_btn = QPushButton("Browse...", self)

        self.cmb_inf_dataset = QComboBox(self)
        self.cmb_inf_dataset.addItem("Validation (val)", "val")
        self.cmb_inf_dataset.addItem("Test (test)", "test")
        self.cmb_inf_dataset.addItem("Train (train)", "train")
        self.cmb_inf_dataset.addItem("Custom NPZ", "custom")

        self.chk_inf_use_future = QCheckBox(
            "Use Stage-1 future NPZ (forecast mode)",
            self,
        )

        self.inf_inputs_edit = QLineEdit(self)
        self.inf_inputs_edit.setPlaceholderText("Custom inputs .npz")
        self.inf_inputs_btn = QPushButton("Inputs...", self)

        self.inf_targets_edit = QLineEdit(self)
        self.inf_targets_edit.setPlaceholderText(
            "Optional targets .npz (for metrics)"
        )
        self.inf_targets_btn = QPushButton("Targets...", self)

        self.sp_inf_batch = QSpinBox(self)
        self.sp_inf_batch.setRange(1, 2048)
        self.sp_inf_batch.setValue(32)

        g1 = QGridLayout()
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
        row.addWidget(self.chk_inf_use_future)
        row.addSpacing(10)
        row.addWidget(QLabel("Batch size:"))
        row.addWidget(self.sp_inf_batch)
        row.addStretch(1)

        g1.addLayout(row, r, 0, 1, 3)
        r += 1

        g1.addWidget(QLabel("Custom inputs:"), r, 0)
        g1.addWidget(self.inf_inputs_edit, r, 1)
        g1.addWidget(self.inf_inputs_btn, r, 2)
        r += 1

        g1.addWidget(QLabel("Custom targets:"), r, 0)
        g1.addWidget(self.inf_targets_edit, r, 1)
        g1.addWidget(self.inf_targets_btn, r, 2)

        self._art_body = QWidget(self)
        self._art_body_l = QVBoxLayout(self._art_body)
        self._art_body_l.setContentsMargins(0, 0, 0, 0)
        self._art_body_l.setSpacing(8)
        self._art_body_l.addLayout(g1)

        b1.addWidget(self._art_body, 0)
        b1.addStretch(1)

        root.addWidget(c1, 0)

        # Card: Uncertainty & outputs
        c2, b2 = self._make_card("Uncertainty & outputs")

        sum2 = QLabel(
            "interval, mode, temperature, calibration",
            self,
        )
        sum2.setObjectName("sumLine")

        self._disc_unc = QToolButton(self)
        self._disc_unc.setObjectName("disclosure")
        self._disc_unc.setText("Edit")
        self._disc_unc.setCheckable(True)
        self._disc_unc.setChecked(True)

        h2 = QHBoxLayout()
        h2.addWidget(sum2, 1)
        h2.addWidget(self._disc_unc, 0)
        b2.addLayout(h2)

        self.chk_inf_use_source_calib = QCheckBox(
            "Use source calibrator (interval_factors_80.npy)",
            self,
        )
        self.chk_inf_fit_calib = QCheckBox(
            "Fit calibrator on validation split",
            self,
        )

        self.inf_calib_edit = QLineEdit(self)
        self.inf_calib_edit.setPlaceholderText(
            "Optional explicit calibrator .npy"
        )
        self.inf_calib_btn = QPushButton("Browse...", self)

        self.sp_inf_cov = QDoubleSpinBox(self)
        self.sp_inf_cov.setDecimals(3)
        self.sp_inf_cov.setRange(0.50, 0.99)
        self.sp_inf_cov.setSingleStep(0.01)
        self.sp_inf_cov.setValue(0.80)

        self.cmb_calib_mode = QComboBox(self)
        modes = CHOICE_SPECS.get("calibration_mode", ("none",))
        for m in modes:
            self.cmb_calib_mode.addItem(m, m)

        self.sp_calib_temp = QDoubleSpinBox(self)
        self.sp_calib_temp.setDecimals(3)
        self.sp_calib_temp.setRange(0.01, 100.0)
        self.sp_calib_temp.setSingleStep(0.10)
        self.sp_calib_temp.setValue(1.0)

        self.sp_cross_pen = QDoubleSpinBox(self)
        self.sp_cross_pen.setDecimals(4)
        self.sp_cross_pen.setRange(0.0, 1e6)
        self.sp_cross_pen.setSingleStep(0.1)
        self.sp_cross_pen.setValue(0.0)

        self.sp_cross_margin = QDoubleSpinBox(self)
        self.sp_cross_margin.setDecimals(4)
        self.sp_cross_margin.setRange(0.0, 1e6)
        self.sp_cross_margin.setSingleStep(0.1)
        self.sp_cross_margin.setValue(0.0)

        self.chk_inf_include_gwl = QCheckBox(
            "Include GWL columns in CSV",
            self,
        )
        self.chk_inf_include_gwl.setChecked(False)

        self.chk_inf_plots = QCheckBox("Generate plots", self)
        self.chk_inf_plots.setChecked(True)

        g2 = QGridLayout()
        c = 0

        row_cal = QHBoxLayout()
        row_cal.addWidget(self.chk_inf_use_source_calib)
        row_cal.addSpacing(10)
        row_cal.addWidget(self.chk_inf_fit_calib)
        row_cal.addStretch(1)
        g2.addLayout(row_cal, c, 0, 1, 3)
        c += 1

        g2.addWidget(QLabel("Calibrator file:"), c, 0)
        g2.addWidget(self.inf_calib_edit, c, 1)
        g2.addWidget(self.inf_calib_btn, c, 2)
        c += 1

        g2.addWidget(QLabel("Interval level:"), c, 0)
        g2.addWidget(self.sp_inf_cov, c, 1, 1, 2)
        c += 1

        g2.addWidget(QLabel("Calibration mode:"), c, 0)
        g2.addWidget(self.cmb_calib_mode, c, 1, 1, 2)
        c += 1

        g2.addWidget(QLabel("Temperature:"), c, 0)
        g2.addWidget(self.sp_calib_temp, c, 1, 1, 2)
        c += 1

        g2.addWidget(QLabel("Cross penalty:"), c, 0)
        g2.addWidget(self.sp_cross_pen, c, 1, 1, 2)
        c += 1

        g2.addWidget(QLabel("Cross margin:"), c, 0)
        g2.addWidget(self.sp_cross_margin, c, 1, 1, 2)
        c += 1

        row_out = QHBoxLayout()
        row_out.addWidget(self.chk_inf_include_gwl)
        row_out.addSpacing(10)
        row_out.addWidget(self.chk_inf_plots)
        row_out.addStretch(1)
        g2.addLayout(row_out, c, 0, 1, 3)

        self._unc_body = QWidget(self)
        self._unc_body_l = QVBoxLayout(self._unc_body)
        self._unc_body_l.setContentsMargins(0, 0, 0, 0)
        self._unc_body_l.setSpacing(8)
        self._unc_body_l.addLayout(g2)

        b2.addWidget(self._unc_body, 0)
        b2.addStretch(1)

        root.addWidget(c2, 0)
        root.addStretch(1)

        scroll.setWidget(page)

        # Hook disclosure toggles
        self._disc_art.toggled.connect(
            lambda v: self._art_body.setVisible(v)
        )
        self._disc_unc.toggled.connect(
            lambda v: self._unc_body.setVisible(v)
        )

        return scroll

    def _build_preview(self) -> QWidget:
        card, body = self._make_card("Run preview")

        self.tree_preview = QTreeWidget(self)
        self.tree_preview.setColumnCount(2)
        self.tree_preview.setHeaderLabels(["Item", "Value"])
        self.tree_preview.setAlternatingRowColors(True)
        self.tree_preview.setRootIsDecorated(True)
        self.tree_preview.setIndentation(14)
        self.tree_preview.setMinimumHeight(240)

        body.addWidget(self.tree_preview, 1)
        body.addStretch(1)

        return card

    def _init_nav_items(self) -> None:
        self._nav_keys = [
            ("artifacts", "Artifacts"),
            ("uncert", "Uncertainty"),
            ("calib", "Calibration"),
            ("outputs", "Outputs"),
            ("adv", "Advanced"),
        ]
        self.nav_list.clear()

        for key, title in self._nav_keys:
            it = QListWidgetItem(self.nav_list)
            it.setData(Qt.UserRole, key)

            w = QWidget(self)
            w.setObjectName("navRow")

            h = QHBoxLayout(w)
            h.setContentsMargins(8, 4, 8, 4)
            h.setSpacing(6)

            t = QLabel(title, self)
            t.setObjectName("navText")

            chip = QLabel("—", self)
            chip.setObjectName("navChip")
            chip.setProperty("status", "off")
            chip.setAlignment(Qt.AlignCenter)

            h.addWidget(t, 1)
            h.addWidget(chip, 0)

            self.nav_list.addItem(it)
            self.nav_list.setItemWidget(it, w)

        if self.nav_list.count() > 0:
            self.nav_list.setCurrentRow(0)

    # ----------------------------------------------------------
    # Wiring
    # ----------------------------------------------------------
    def _connect_ui(self) -> None:
        self.btn_run_infer.clicked.connect(self.run_clicked.emit)
        self.btn_inf_options.clicked.connect(
            self.advanced_clicked.emit
        )

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

        self._btn_refresh.clicked.connect(self._update_preview)

        self.cmb_inf_dataset.currentIndexChanged.connect(
            self._update_widgets_state
        )
        self.chk_inf_use_future.toggled.connect(
            self._update_widgets_state
        )

        for w in (
            self.inf_model_edit,
            self.inf_manifest_edit,
            self.inf_inputs_edit,
            self.inf_targets_edit,
            self.inf_calib_edit,
        ):
            w.textChanged.connect(self._update_preview)

        for w in (
            self.chk_inf_use_source_calib,
            self.chk_inf_fit_calib,
            self.chk_inf_include_gwl,
            self.chk_inf_plots,
        ):
            w.toggled.connect(self._update_preview)

        self.sp_inf_batch.valueChanged.connect(
            lambda _v: self._update_preview()
        )

        self.btn_open_last_dir.clicked.connect(self._open_last_dir)
        self.btn_open_last_eval.clicked.connect(
            self._open_last_eval
        )
        self.btn_open_last_future.clicked.connect(
            self._open_last_future
        )
        self.btn_open_last_json.clicked.connect(
            self._open_last_json
        )

        self.sp_inf_cov.valueChanged.connect(
            self._on_interval_level_changed
        )
        self.cmb_calib_mode.currentIndexChanged.connect(
            self._on_calib_mode_changed
        )
        self.sp_calib_temp.valueChanged.connect(
            self._on_calib_temp_changed
        )
        self.sp_cross_pen.valueChanged.connect(
            self._on_cross_pen_changed
        )
        self.sp_cross_margin.valueChanged.connect(
            self._on_cross_margin_changed
        )

        self.nav_list.currentRowChanged.connect(
            self._on_nav_selected
        )

    def _bind_store(self) -> None:
        self._store.config_changed.connect(self._on_store_changed)

    def _on_store_changed(self, keys: object) -> None:
        _ = keys
        self._sync_ui_from_store()
        self._update_preview()
        self._refresh_nav_chips()

    # ----------------------------------------------------------
    # Store sync (store-driven fields)
    # ----------------------------------------------------------
    def _sync_ui_from_store(self) -> None:
        s = self._store

        with QSignalBlocker(self.sp_inf_cov):
            v = s.get_value(
                FieldKey("interval_level"),
                default=0.80,
            )
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

    def _store_set(self, name: str, value: Any) -> None:
        try:
            self._store.set_value_by_key(FieldKey(name), value)
        except Exception:
            return

    def _on_interval_level_changed(self, v: float) -> None:
        self._store_set("interval_level", float(v))

    def _on_calib_mode_changed(self, _i: int) -> None:
        v = self.cmb_calib_mode.currentData() or "none"
        self._store_set("calibration_mode", str(v))
        self._update_preview()

    def _on_calib_temp_changed(self, v: float) -> None:
        self._store_set("calibration_temperature", float(v))

    def _on_cross_pen_changed(self, v: float) -> None:
        self._store_set("crossing_penalty", float(v))

    def _on_cross_margin_changed(self, v: float) -> None:
        self._store_set("crossing_margin", float(v))

    # ----------------------------------------------------------
    # UI state rules
    # ----------------------------------------------------------
    def _update_widgets_state(self) -> None:
        key = self.cmb_inf_dataset.currentData() or "test"
        use_future = self.chk_inf_use_future.isChecked()

        custom = (key == "custom") and (not use_future)

        self.inf_inputs_edit.setEnabled(custom)
        self.inf_inputs_btn.setEnabled(custom)

        self.inf_targets_edit.setEnabled(custom)
        self.inf_targets_btn.setEnabled(custom)

        self._update_preview()

    # ----------------------------------------------------------
    # Navigator + chips
    # ----------------------------------------------------------
    def _refresh_nav_chips(self) -> None:
        chips = compute_infer_nav(self._store)

        for i in range(self.nav_list.count()):
            it = self.nav_list.item(i)
            key = str(it.data(Qt.UserRole) or "")
            w = self.nav_list.itemWidget(it)
            if w is None:
                continue

            chip = w.findChild(QLabel, "navChip")
            if chip is None:
                continue

            meta = chips.get(key, {"status": "off", "text": "—"})
            chip.setText(str(meta.get("text", "—")))
            chip.setProperty("status", meta.get("status", "off"))

            chip.style().unpolish(chip)
            chip.style().polish(chip)

    def _on_nav_selected(self, row: int) -> None:
        for i in range(self.nav_list.count()):
            it = self.nav_list.item(i)
            w = self.nav_list.itemWidget(it)
            if w is None:
                continue

            w.setProperty("selected", i == row)
            w.style().unpolish(w)
            w.style().polish(w)

        # Later: scroll to matching card.

    # ----------------------------------------------------------
    # Last outputs
    # ----------------------------------------------------------
    def _sync_last_outputs_ui(self) -> None:
        out_dir = self._last_outputs.get("run_dir", "")
        csv_eval = self._last_outputs.get("csv_eval_path", "")
        csv_fut = self._last_outputs.get("csv_future_path", "")
        js = self._last_outputs.get("inference_summary_json", "")

        self.ed_last_dir.setText(out_dir)
        self.ed_last_eval.setText(csv_eval)
        self.ed_last_future.setText(csv_fut)
        self.ed_last_json.setText(js)

        self.btn_open_last_dir.setEnabled(_exists(out_dir))
        self.btn_open_last_eval.setEnabled(_exists(csv_eval))
        self.btn_open_last_future.setEnabled(_exists(csv_fut))
        self.btn_open_last_json.setEnabled(_exists(js))

    def _open_last_dir(self) -> None:
        p = self.ed_last_dir.text().strip()
        if _exists(p):
            QDesktopServices.openUrl(QUrl.fromLocalFile(p))

    def _open_last_eval(self) -> None:
        p = self.ed_last_eval.text().strip()
        if _exists(p):
            QDesktopServices.openUrl(QUrl.fromLocalFile(p))

    def _open_last_future(self) -> None:
        p = self.ed_last_future.text().strip()
        if _exists(p):
            QDesktopServices.openUrl(QUrl.fromLocalFile(p))

    def _open_last_json(self) -> None:
        p = self.ed_last_json.text().strip()
        if _exists(p):
            QDesktopServices.openUrl(QUrl.fromLocalFile(p))

    # ----------------------------------------------------------
    # Preview
    # ----------------------------------------------------------
    def _pv_section(self, title: str) -> QTreeWidgetItem:
        item = QTreeWidgetItem([title, ""])
        item.setFirstColumnSpanned(True)

        f = QFont(item.font(0))
        f.setBold(True)
        item.setFont(0, f)

        self.tree_preview.addTopLevelItem(item)
        item.setExpanded(True)
        return item

    def _pv_kv(
        self,
        parent: QTreeWidgetItem,
        key: str,
        val: str,
    ) -> None:
        QTreeWidgetItem(parent, [key, val])

    def _update_preview(self) -> None:
        self._plan_sum.setText(build_plan_text(self._store))

        model_p = self.inf_model_edit.text().strip()
        mani_p = self.inf_manifest_edit.text().strip()

        dkey = self.cmb_inf_dataset.currentData() or "test"
        use_future = self.chk_inf_use_future.isChecked()

        inputs_p = self.inf_inputs_edit.text().strip()
        targets_p = self.inf_targets_edit.text().strip()

        cov = float(self.sp_inf_cov.value())
        mode = self.cmb_calib_mode.currentData() or "none"
        temp = float(self.sp_calib_temp.value())

        src_cal = self.chk_inf_use_source_calib.isChecked()
        fit_cal = self.chk_inf_fit_calib.isChecked()
        cal_p = self.inf_calib_edit.text().strip()

        inc_gwl = self.chk_inf_include_gwl.isChecked()
        plots = self.chk_inf_plots.isChecked()
        bsz = int(self.sp_inf_batch.value())

        run_dir = ""
        if model_p:
            mp = _abspath(model_p)
            run_dir = mp if os.path.isdir(mp) else os.path.dirname(mp)

        warn: list[str] = []

        if not model_p:
            warn.append("- Missing model path.")
        elif not _exists(model_p):
            warn.append("- Model path does not exist.")

        if (dkey == "custom") and (not use_future):
            if not inputs_p:
                warn.append("- Custom inputs NPZ is required.")
            elif not _exists(inputs_p):
                warn.append("- Inputs NPZ does not exist.")

            if targets_p and (not _exists(targets_p)):
                warn.append("- Targets NPZ does not exist.")

        if mani_p and (not _exists(mani_p)):
            warn.append("- Manifest path does not exist.")

        if cal_p and (not _exists(cal_p)):
            warn.append("- Calibrator file does not exist.")

        self.tree_preview.clear()

        ready = "OK" if not warn else "Needs attention"
        sec = self._pv_section("Readiness")
        self._pv_kv(sec, "Status", ready)
        self._pv_kv(sec, "Warnings", str(len(warn)))

        sec = self._pv_section("Inputs")
        self._pv_kv(sec, "Dataset", f"{dkey} (future={use_future})")
        self._pv_kv(sec, "Batch size", str(bsz))
        self._pv_kv(sec, "Model", _abspath(model_p) or "(empty)")
        self._pv_kv(
            sec,
            "Stage-1 manifest",
            _abspath(mani_p) or "(auto)",
        )
        self._pv_kv(sec, "Run dir", run_dir or "(unknown)")

        sec = self._pv_section("Uncertainty (store)")
        self._pv_kv(sec, "Interval level", f"{cov:.3f}")
        self._pv_kv(sec, "Mode", str(mode))
        self._pv_kv(sec, "Temperature", f"{temp:.3f}")
        self._pv_kv(
            sec,
            "Cross penalty",
            f"{self.sp_cross_pen.value():.4f}",
        )
        self._pv_kv(
            sec,
            "Cross margin",
            f"{self.sp_cross_margin.value():.4f}",
        )

        sec = self._pv_section("Calibration (runtime)")
        self._pv_kv(sec, "Use source", str(src_cal))
        self._pv_kv(sec, "Fit on val", str(fit_cal))
        self._pv_kv(sec, "Explicit file", _abspath(cal_p) or "(none)")

        sec = self._pv_section("Outputs (runtime)")
        self._pv_kv(sec, "Include GWL", str(inc_gwl))
        self._pv_kv(sec, "Generate plots", str(plots))

        if warn:
            sec = self._pv_section("Warnings")
            for w in warn:
                self._pv_kv(sec, "", w)

        self._lbl_status.setText(ready)
