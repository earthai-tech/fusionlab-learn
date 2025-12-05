# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>

from __future__ import annotations

from typing import Mapping, Any, Optional

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QDialog,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGroupBox,
    QGridLayout,
    QLabel,
    QDoubleSpinBox,
    QComboBox,
    QPushButton,
    QDialogButtonBox,
)


PHYSICS_DEFAULTS = {
    "MV_LR_MULT": 1.0,
    "KAPPA_LR_MULT": 5.0,
    "GEOPRIOR_INIT_MV": 1e-7,
    "GEOPRIOR_INIT_KAPPA": 1.0,
    "GEOPRIOR_GAMMA_W": 9810.0,
    "GEOPRIOR_H_REF": 0.0,
    "GEOPRIOR_KAPPA_MODE": "kb",  # {"bar", "kb"}
    "GEOPRIOR_HD_FACTOR": 0.6,
}


class PhysicsConfigDialog(QDialog):
    """
    Configure scalar physics parameters used by GeoPriorSubsNet.

    The dialog exposes:

    * Learning-rate multipliers for scalar PINN parameters
      (``MV_LR_MULT``, ``KAPPA_LR_MULT``).
    * GeoPrior scalar parameters
      (``GEOPRIOR_INIT_MV``, ``GEOPRIOR_INIT_KAPPA``,
       ``GEOPRIOR_GAMMA_W``, ``GEOPRIOR_H_REF``,
       ``GEOPRIOR_KAPPA_MODE``, ``GEOPRIOR_HD_FACTOR``).

    Values are returned as a flat dict with the same keys.
    """

    def __init__(
        self,
        parent: Optional[QWidget] = None,
        *,
        initial: Optional[Mapping[str, Any]] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Physics configuration")
        self.setModal(True)

        cfg = dict(PHYSICS_DEFAULTS)
        if initial:
            cfg.update({k: v for k, v in initial.items() if k in cfg})

        main = QVBoxLayout(self)
        info = QLabel(
            "Configure learning-rate multipliers and scalar GeoPrior "
            "parameters. These values control how the PINN physics "
            "block is initialised and optimised."
        )
        info.setWordWrap(True)
        main.addWidget(info)

        # ------------------------------------------------------------------
        # GeoPrior scalar parameters group
        # ------------------------------------------------------------------
        gp_group = QGroupBox("GeoPrior scalar parameters")
        gp_grid = QGridLayout(gp_group)
        gp_grid.setHorizontalSpacing(16)
        gp_grid.setVerticalSpacing(6)

        # Spin boxes + combo
        self.init_mv_spin = self._make_spinbox(
            minimum=1e-10,
            maximum=1e-2,
            step=1e-7,
            decimals=8,
            value=float(cfg["GEOPRIOR_INIT_MV"]),
        )
        self.init_kappa_spin = self._make_spinbox(
            minimum=1e-3,
            maximum=1e2,
            step=0.01,
            decimals=4,
            value=float(cfg["GEOPRIOR_INIT_KAPPA"]),
        )
        self.gamma_w_spin = self._make_spinbox(
            minimum=0.0,
            maximum=30000.0,
            step=10.0,
            decimals=3,
            value=float(cfg["GEOPRIOR_GAMMA_W"]),
        )
        self.h_ref_spin = self._make_spinbox(
            minimum=-1000.0,
            maximum=1000.0,
            step=0.1,
            decimals=3,
            value=float(cfg["GEOPRIOR_H_REF"]),
        )
        self.hd_factor_spin = self._make_spinbox(
            minimum=0.0,
            maximum=1.0,
            step=0.01,
            decimals=3,
            value=float(cfg["GEOPRIOR_HD_FACTOR"]),
        )

        self.kappa_mode_combo = QComboBox()
        self.kappa_mode_combo.addItems(["bar", "kb"])
        self.kappa_mode_combo.setMaximumWidth(130)
        current_mode = str(cfg["GEOPRIOR_KAPPA_MODE"]).lower()
        idx = self.kappa_mode_combo.findText(current_mode)
        if idx >= 0:
            self.kappa_mode_combo.setCurrentIndex(idx)

        # ---- Left column: material priors (mᵥ, κ̄) -------------------
        row = 0
        lbl_mv = QLabel("Initial compressibility mᵥ [Pa⁻¹]:")
        lbl_mv.setToolTip(
            "Prior value for the vertical constrained compressibility mᵥ.\n"
            "Controls how much vertical strain is produced per unit change\n"
            "in effective stress; enters s_eq = mᵥ γw Δh H."
        )
        gp_grid.addWidget(lbl_mv, row, 0)
        gp_grid.addWidget(self.init_mv_spin, row, 1)

        lbl_kappa = QLabel("Initial closure factor κ̄ [–]:")
        lbl_kappa.setToolTip(
            "Prior dimensionless factor κ̄ (κb / anisotropy factor) used in\n"
            "the timescale closure linking horizontal K and vertical kv."
        )
        gp_grid.addWidget(lbl_kappa, row, 2)
        gp_grid.addWidget(self.init_kappa_spin, row, 3)
        row += 1

        # ---- Middle row: hydrostatic environment (γw, href) ----------
        lbl_gamma = QLabel("Unit weight of water γw [N m⁻³]:")
        lbl_gamma.setToolTip(
            "Unit weight of water converting head change Δh to effective\n"
            "stress change Δσ' = γw Δh; appears in Cv = kv/(mᵥ γw)."
        )
        gp_grid.addWidget(lbl_gamma, row, 0)
        gp_grid.addWidget(self.gamma_w_spin, row, 1)

        lbl_href = QLabel("Reference head href [m]:")
        lbl_href.setToolTip(
            "Reference groundwater head defining drawdown Δh = href − h̄.\n"
            "Used in the prior for equilibrium settlement s_eq( h̄ )."
        )
        gp_grid.addWidget(lbl_href, row, 2)
        gp_grid.addWidget(self.h_ref_spin, row, 3)
        row += 1

        # ---- Bottom row: closure mode + drainage factor ---------------
        lbl_mode = QLabel("κ closure mode (bar / kb):")
        lbl_mode.setToolTip(
            "Choice of parameterisation for the κ closure:\n"
            "  • 'bar' → κ̄ in the prior;\n"
            "  • 'kb' → kb factor in kv = kb K."
        )
        gp_grid.addWidget(lbl_mode, row, 0)
        gp_grid.addWidget(self.kappa_mode_combo, row, 1)

        lbl_hd = QLabel("Drainage-thickness factor Hd/H [–]:")
        lbl_hd.setToolTip(
            "Dimensionless factor relating drainage path Hd to effective\n"
            "compressible thickness H; enters the consolidation timescale\n"
            "τ = Hd² / (π² Cv) via the mapping note κ̄ = (Hd/H)² / κb."
        )
        gp_grid.addWidget(lbl_hd, row, 2)
        gp_grid.addWidget(self.hd_factor_spin, row, 3)

        main.addWidget(gp_group)

        # ------------------------------------------------------------------
        # Learning-rate multipliers group (now *below* GeoPrior parameters)
        # ------------------------------------------------------------------
        lr_group = QGroupBox("Learning-rate multipliers")
        lr_grid = QGridLayout(lr_group)
        lr_grid.setHorizontalSpacing(16)
        lr_grid.setVerticalSpacing(4)

        self.mv_lr_spin = self._make_spinbox(
            minimum=0.0,
            maximum=100.0,
            step=0.1,
            decimals=3,
            value=float(cfg["MV_LR_MULT"]),
        )
        self.kappa_lr_spin = self._make_spinbox(
            minimum=0.0,
            maximum=100.0,
            step=0.1,
            decimals=3,
            value=float(cfg["KAPPA_LR_MULT"]),
        )

        # ---- One row: mᵥ LR  |  κ̄ LR ---------------------------------
        lbl_mv_lr = QLabel("mᵥ LR multiplier [–]:")
        lbl_mv_lr.setToolTip(
            "Dimensionless factor scaling the base learning rate for the\n"
            "vertical compressibility mᵥ. Effective step size is\n"
            "lr_mᵥ = factor × base learning rate."
        )
        lr_grid.addWidget(lbl_mv_lr, 0, 0)
        lr_grid.addWidget(self.mv_lr_spin, 0, 1)

        lbl_kappa_lr = QLabel("κ̄ LR multiplier [–]:")
        lbl_kappa_lr.setToolTip(
            "Dimensionless factor scaling the base learning rate for the\n"
            "closure / leakage factor κ̄. Controls how fast κ̄ adapts\n"
            "relative to the rest of the network weights."
        )
        lr_grid.addWidget(lbl_kappa_lr, 0, 2)
        lr_grid.addWidget(self.kappa_lr_spin, 0, 3)

        main.addWidget(lr_group)
        
        # ------------------------------------------------------------------
        # Buttons: Reset / OK / Cancel
        # ------------------------------------------------------------------
        btn_row = QHBoxLayout()
        self.reset_btn = QPushButton("Reset defaults")
        self.reset_btn.clicked.connect(self._reset_defaults)
        btn_row.addWidget(self.reset_btn)
        btn_row.addStretch(1)

        button_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        btn_row.addWidget(button_box)

        main.addLayout(btn_row)

        # Slightly wider than before, but compact vertically
        self.resize(520, self.sizeHint().height())

    # ------------------------------------------------------------------ #
    # Helpers                                                            #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _make_spinbox(
        *,
        minimum: float,
        maximum: float,
        step: float,
        decimals: int,
        value: float,
    ) -> QDoubleSpinBox:
        sb = QDoubleSpinBox()
        sb.setRange(minimum, maximum)
        sb.setDecimals(decimals)
        sb.setSingleStep(step)
        sb.setValue(value)
        sb.setAlignment(Qt.AlignRight)
        # keep narrow so labels dominate width
        sb.setMaximumWidth(130)
        return sb

    def _reset_defaults(self) -> None:
        """Restore all fields to `PHYSICS_DEFAULTS`."""
        self.mv_lr_spin.setValue(PHYSICS_DEFAULTS["MV_LR_MULT"])
        self.kappa_lr_spin.setValue(PHYSICS_DEFAULTS["KAPPA_LR_MULT"])
        self.init_mv_spin.setValue(PHYSICS_DEFAULTS["GEOPRIOR_INIT_MV"])
        self.init_kappa_spin.setValue(PHYSICS_DEFAULTS["GEOPRIOR_INIT_KAPPA"])
        self.gamma_w_spin.setValue(PHYSICS_DEFAULTS["GEOPRIOR_GAMMA_W"])
        self.h_ref_spin.setValue(PHYSICS_DEFAULTS["GEOPRIOR_H_REF"])
        self.hd_factor_spin.setValue(PHYSICS_DEFAULTS["GEOPRIOR_HD_FACTOR"])
        mode = PHYSICS_DEFAULTS["GEOPRIOR_KAPPA_MODE"]
        idx = self.kappa_mode_combo.findText(str(mode).lower())
        if idx >= 0:
            self.kappa_mode_combo.setCurrentIndex(idx)

    # ------------------------------------------------------------------ #
    # Public API                                                         #
    # ------------------------------------------------------------------ #
    def to_dict(self) -> dict:
        """Return current values as a flat NAT-style dict."""
        return {
            "MV_LR_MULT": float(self.mv_lr_spin.value()),
            "KAPPA_LR_MULT": float(self.kappa_lr_spin.value()),
            "GEOPRIOR_INIT_MV": float(self.init_mv_spin.value()),
            "GEOPRIOR_INIT_KAPPA": float(self.init_kappa_spin.value()),
            "GEOPRIOR_GAMMA_W": float(self.gamma_w_spin.value()),
            "GEOPRIOR_H_REF": float(self.h_ref_spin.value()),
            "GEOPRIOR_KAPPA_MODE": str(
                self.kappa_mode_combo.currentText()
            ).lower(),
            "GEOPRIOR_HD_FACTOR": float(self.hd_factor_spin.value()),
        }

    @staticmethod
    def edit_physics(
        parent: QWidget,
        cfg: Mapping[str, Any],
    ) -> Optional[dict]:
        """
        Convenience helper: open the dialog and return an updated
        config dict (or ``None`` if cancelled).
        """
        initial = {k: cfg.get(k, v) for k, v in PHYSICS_DEFAULTS.items()}
        dlg = PhysicsConfigDialog(parent=parent, initial=initial)
        if dlg.exec_() == QDialog.Accepted:
            updated = dict(cfg)
            updated.update(dlg.to_dict())
            return updated
        return None
