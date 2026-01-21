# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple

from PyQt5.QtCore import Qt, QSignalBlocker, pyqtSignal
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
    QSizePolicy  
)
from PyQt5.QtGui import QFontMetrics

from ...config.store import GeoConfigStore
from ...config.prior_schema import FieldKey
from .lifecycle import TrainingLifecycle


MakeCardFn = Callable[[str], Tuple[QWidget, QVBoxLayout]]
MakeRunBtnFn = Callable[[str], QPushButton]


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

        self._build_ui()
        self._wire_ui()
        self._wire_store()
        self.refresh_from_store()

    # -----------------------------------------------------------------
    # UI build
    # -----------------------------------------------------------------
    
    def _build_ui(self) -> None:
        # =========================================================
        # Root layout
        # =========================================================
        t_layout = QVBoxLayout(self)
        t_layout.setContentsMargins(6, 6, 6, 6)
        t_layout.setSpacing(8)
    
        def _field(w: QWidget) -> None:
            w.setSizePolicy(
                QSizePolicy.Expanding,
                QSizePolicy.Fixed,
            )
            w.setMinimumHeight(26)
            w.setMinimumWidth(140)
    
        def _btn(b: QPushButton) -> None:
            b.setSizePolicy(
                QSizePolicy.Expanding,
                QSizePolicy.Fixed,
            )
            b.setMinimumHeight(30)
    
        right_min_w = 280
    
        # =========================================================
        # Card: Temporal window
        # =========================================================
        temp_card, temp_box = self._make_card("Temporal window")
        temp_card.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Preferred,
        )
    
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
    
        temp_box.addLayout(grid_t)
        temp_box.addStretch(1)
    
        # =========================================================
        # Card: Training
        # =========================================================
        train_card, train_box = self._make_card("Training")
        train_card.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Preferred,
        )
    
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
    
        train_box.addLayout(grid_tr)
        train_box.addStretch(1)
    
        # =========================================================
        # Card: Physics weights
        # =========================================================
        phys_w_card, phys_w_box = self._make_card("Physics weights")
        phys_w_card.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Preferred,
        )
        phys_w_card.setMinimumWidth(right_min_w)
    
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
    
        phys_w_box.addLayout(grid_w)
        phys_w_box.addStretch(1)
    
        # =========================================================
        # Card: Quick actions (spans rows 2-3, cols 1-2)
        # =========================================================
        # =========================================================
        # Card: Quick actions (buttons + preset only)
        # =========================================================
        quick_card, quick_box = self._make_card("Quick actions")
        quick_card.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Preferred,
        )

        btn_grid = QGridLayout()
        btn_grid.setHorizontalSpacing(10)
        btn_grid.setVerticalSpacing(8)
        btn_grid.setColumnStretch(0, 1)
        btn_grid.setColumnStretch(1, 1)

        self.btn_features = QPushButton("Feature config...")
        self.btn_arch = QPushButton("Architecture...")
        self.btn_prob = QPushButton("Probabilistic...")
        self.physics_btn = QPushButton("Physics config...")

        for b in (
            self.btn_features,
            self.btn_arch,
            self.btn_prob,
            self.physics_btn,
        ):
            _btn(b)

        btn_grid.addWidget(self.btn_features, 0, 0)
        btn_grid.addWidget(self.btn_arch, 0, 1)
        btn_grid.addWidget(self.btn_prob, 1, 0)
        btn_grid.addWidget(self.physics_btn, 1, 1)

        quick_box.addLayout(btn_grid)
        quick_box.addSpacing(10)

        preset_row = QHBoxLayout()
        preset_row.setSpacing(8)
        preset_row.addWidget(QLabel("Preset:"))

        self.cmb_preset = QComboBox()
        self.cmb_preset.addItems(list(self._presets.keys()))
        self.cmb_preset.setCurrentText("Custom")
        _field(self.cmb_preset)

        preset_row.addWidget(self.cmb_preset, 1)
        quick_box.addLayout(preset_row)
        quick_box.addStretch(1)

        self.chk_clean_stage1 = QCheckBox("Clean Stage-1 run dir")
        self.chk_clean_stage1.setVisible(False)

        # =========================================================
        # Card: Lifecycle & base model (own section)
        # =========================================================
        life_card, life_box = self._make_card("Training initialization")
        life_card.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Expanding,
        )

        self.life = TrainingLifecycle(store=self._store)
        life_box.addWidget(self.life, 1)
        life_box.addStretch(1)

        # =========================================================
        # Left stack widget (Quick actions on top, lifecycle below)
        # Spans rows 2-3, cols 1-2
        # =========================================================
        left_stack = QWidget(self)
        left_stack.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Expanding,
        )

        left_l = QVBoxLayout(left_stack)
        left_l.setContentsMargins(0, 0, 0, 0)
        left_l.setSpacing(10)

        left_l.addWidget(quick_card, 0)
        left_l.addWidget(life_card, 1)

    
        # =========================================================
        # Card: Physics schedule (right col, row 2)
        # Modern: compact controls + good spacing
        # =========================================================
        phys_s_card, phys_s_box = self._make_card("Physics schedule")
        phys_s_card.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Expanding,
        )
        phys_s_card.setMinimumWidth(right_min_w)
    
        self.sp_phys_warmup = QSpinBox()
        self.sp_phys_warmup.setRange(0, 10_000_000)
        _field(self.sp_phys_warmup)
    
        self.sp_phys_ramp = QSpinBox()
        self.sp_phys_ramp.setRange(0, 10_000_000)
        _field(self.sp_phys_ramp)
    
        self.chk_scale_pde = QCheckBox(
            "Scale PDE residuals (stable gradients)"
        )
    
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
    
        phys_s_box.addLayout(grid_s)
        phys_s_box.addStretch(1)
    
        # =========================================================
        # Card: Training options (right col, row 3)
        # Modern: full-width "Advanced" + clean grouping
        # =========================================================
        opts_card, opts_box = self._make_card("Training options")
        opts_card.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Expanding,
        )
        opts_card.setMinimumWidth(right_min_w)
    
        self.chk_eval_training = QCheckBox(
            "Evaluate training metrics"
        )
        self.chk_build_future = QCheckBox("Build future NPZ")
    
        self.btn_train_options = QPushButton("Advanced options...")
        _btn(self.btn_train_options)
    
        opts_box.addWidget(self.chk_eval_training)
        opts_box.addWidget(self.chk_build_future)
        opts_box.addSpacing(10)
        opts_box.addWidget(self.btn_train_options)
        opts_box.addStretch(1)
    
        # =========================================================
        # No "Run" card.
        # Keep: status (left) + "Run:" + [button] bottom-right
        # =========================================================
        self.lbl_run_status = QLabel("Ready.")
        self.lbl_run_status.setWordWrap(False)
        self.lbl_run_status.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Fixed,
        )
        self.lbl_run_status.setMinimumHeight(18)
        self.lbl_run_status.setMaximumHeight(18)
        
        # You can keep "Run training" if you want.
        # If you want it minimal, use "Run".
        self.train_btn = self._make_run_button("Run")
        self.train_btn.setMinimumHeight(38)
        self.train_btn.setObjectName("runButton")
        self.train_btn.setSizePolicy(
            QSizePolicy.Minimum,
            QSizePolicy.Fixed,
        )
        
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

    
        # =========================================================
        # Main grid: expected layout (4 rows x 3 cols)
        # =========================================================
        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(10)
    
        # Row 1
        grid.addWidget(temp_card, 0, 0)
        grid.addWidget(train_card, 0, 1)
        grid.addWidget(phys_w_card, 0, 2)
    
        # Rows 2-3 (Quick spans 2 rows, 2 cols)
        # grid.addWidget(quick_card, 1, 0, 2, 2)
        grid.addWidget(left_stack, 1, 0, 2, 2)
    
        # Right column stack
        # Right column stack
        grid.addWidget(phys_s_card, 1, 2)
        grid.addWidget(opts_card, 2, 2)
        
        # Bottom row: status spans col0-col1,
        # run controls in col2 (bottom-right).
        grid.addWidget(self.lbl_run_status, 3, 0, 1, 2)
        grid.addLayout(run_row, 3, 2)
        
        # Row sizing (top compact, middle expands, bottom compact)
        grid.setRowStretch(0, 0)
        grid.setRowStretch(1, 1)
        grid.setRowStretch(2, 1)
        grid.setRowStretch(3, 0)

        t_layout.addLayout(grid, 1)

    
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

    def _on_preset_changed(self) -> None:
        name = str(self.cmb_preset.currentText()).strip()
        if not name or name == "Custom":
            return
        self._apply_preset(name)

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
        # Bind widgets to store keys (GeoPriorConfig fields)
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

        # Emit actions (app connects to these)
        self.train_btn.clicked.connect(self.run_clicked)
        self.btn_train_options.clicked.connect(
            self.advanced_clicked
        )
        self.btn_features.clicked.connect(self.features_clicked)
        self.btn_arch.clicked.connect(self.arch_clicked)
        self.btn_prob.clicked.connect(self.prob_clicked)
        self.physics_btn.clicked.connect(self.physics_clicked)
        
        self.cmb_preset.currentIndexChanged.connect(
            lambda _=0: self._on_preset_changed()
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
                # Never let UI crash on a bad value
                continue
            
        if hasattr(self, "life"):
            self.life.refresh_from_store()

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
