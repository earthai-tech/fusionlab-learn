# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>

"""geoprior.ui.setup.cards.model_architecture

Model architecture card (modern UX).

- Model selector + roadmap messages.
- Core knobs shown on-card.
- Full attention config via ArchitectureConfigDialog.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from PyQt5.QtCore import Qt, QSignalBlocker
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QStyle,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from .base import CardBase
from ..bindings import Binder
from ....config.store import GeoConfigStore
from ....dialogs.architecture_dialog import (
    ArchitectureConfigDialog,
)


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


_MODEL_ITEMS = [
    ("GeoPriorSubsNet (default)", "GeoPriorSubsNet"),
    ("PoroElasticSubsNet (next)", "PoroElasticSubsNet"),
    ("HybridAttn (future)", "HybridAttn"),
]


class ModelArchitectureCard(CardBase):
    """Model architecture card (store-driven)."""

    def __init__(
        self,
        *,
        store: GeoConfigStore,
        binder: Binder,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(
            section_id="arch",
            title="Model architecture",
            subtitle=(
                "Select a model backbone and adjust core "
                "dimensions. Advanced attention settings "
                "are available in the Architecture dialog."
            ),
            parent=parent,
        )

        self.store = store
        self.binder = binder

        self._build()
        self._wire()
        self._sync_preview()

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

        self.left = QWidget(grid)
        l = QVBoxLayout(self.left)
        l.setContentsMargins(0, 0, 0, 0)
        l.setSpacing(10)

        self.right = QWidget(grid)
        r = QVBoxLayout(self.right)
        r.setContentsMargins(0, 0, 0, 0)
        r.setSpacing(10)

        self.grp_model = self._build_model_box(self.left)
        self.grp_core = self._build_core_box(self.left)
        self.grp_flags = self._build_flags_box(self.left)
        self.grp_adv = self._build_advanced_box(self.left)

        l.addWidget(self.grp_model, 0)
        l.addWidget(self.grp_core, 0)
        l.addWidget(self.grp_flags, 0)
        l.addWidget(self.grp_adv, 0)
        l.addStretch(1)

        self.grp_preview = self._build_preview_box(self.right)
        r.addWidget(self.grp_preview, 0)
        r.addStretch(1)

        g.addWidget(self.left, 0, 0)
        g.addWidget(self.right, 0, 1)

        body.addWidget(grid, 0)

        self.add_action(
            text="Configure…",
            tip="Open full architecture configuration.",
            icon=QStyle.SP_FileDialogDetailedView,
        ).clicked.connect(self._open_arch_dialog)

    def _build_model_box(self, parent: QWidget) -> QGroupBox:
        box = QGroupBox("Model", parent)
        lay = QGridLayout(box)
        lay.setContentsMargins(10, 10, 10, 10)
        lay.setHorizontalSpacing(10)
        lay.setVerticalSpacing(8)
        lay.setColumnStretch(1, 1)

        self.cmb_model = QComboBox(box)
        self.binder.bind_combo(
            "model_name",
            self.cmb_model,
            items=_MODEL_ITEMS,
            editable=False,
            use_item_data=True,
        )

        self.lbl_banner = QLabel("", box)
        self.lbl_banner.setWordWrap(True)
        self.lbl_banner.setVisible(False)
        self.lbl_banner.setObjectName("archBanner")

        box.setStyleSheet(
            "\n".join(
                [
                    "QLabel#archBanner {",
                    "  padding: 6px 8px;",
                    "  border-radius: 8px;",
                    "  border: 1px solid",
                    "    rgba(242,134,32,0.40);",
                    "  background: rgba(242,134,32,0.12);",
                    "  color: rgba(30,30,30,0.86);",
                    "  font-size: 11px;",
                    "}",
                ]
            )
        )

        lay.addWidget(QLabel("Backbone:", box), 0, 0)
        lay.addWidget(self.cmb_model, 0, 1)
        lay.addWidget(self.lbl_banner, 1, 0, 1, 2)

        return box

    def _build_core_box(self, parent: QWidget) -> QGroupBox:
        box = QGroupBox("Core dimensions", parent)
        lay = QGridLayout(box)
        lay.setContentsMargins(10, 10, 10, 10)
        lay.setHorizontalSpacing(10)
        lay.setVerticalSpacing(8)
        lay.setColumnStretch(1, 1)
        lay.setColumnStretch(3, 1)

        def _spin(
            minv: int,
            maxv: int,
        ) -> QSpinBox:
            sp = QSpinBox(box)
            sp.setRange(minv, maxv)
            return sp

        def _dspin(
            minv: float,
            maxv: float,
            step: float,
        ) -> QDoubleSpinBox:
            sp = QDoubleSpinBox(box)
            sp.setDecimals(3)
            sp.setRange(minv, maxv)
            sp.setSingleStep(step)
            return sp

        self.sp_embed = _spin(8, 512)
        self.sp_hidden = _spin(8, 1024)
        self.sp_lstm = _spin(8, 1024)
        self.sp_att = _spin(8, 512)
        self.sp_heads = _spin(1, 16)
        self.sp_drop = _dspin(0.0, 0.90, 0.01)

        self.binder.bind_spin_box("embed_dim", self.sp_embed)
        self.binder.bind_spin_box(
            "hidden_units",
            self.sp_hidden,
        )
        self.binder.bind_spin_box("lstm_units", self.sp_lstm)
        self.binder.bind_spin_box(
            "attention_units",
            self.sp_att,
        )
        self.binder.bind_spin_box("num_heads", self.sp_heads)
        self.binder.bind_double_spin_box(
            "dropout_rate",
            self.sp_drop,
        )

        r = 0
        lay.addWidget(QLabel("Embed:", box), r, 0)
        lay.addWidget(self.sp_embed, r, 1)
        lay.addWidget(QLabel("Hidden:", box), r, 2)
        lay.addWidget(self.sp_hidden, r, 3)
        r += 1

        lay.addWidget(QLabel("LSTM:", box), r, 0)
        lay.addWidget(self.sp_lstm, r, 1)
        lay.addWidget(QLabel("Attention:", box), r, 2)
        lay.addWidget(self.sp_att, r, 3)
        r += 1

        lay.addWidget(QLabel("Heads:", box), r, 0)
        lay.addWidget(self.sp_heads, r, 1)
        lay.addWidget(QLabel("Dropout:", box), r, 2)
        lay.addWidget(self.sp_drop, r, 3)

        return box

    def _build_flags_box(self, parent: QWidget) -> QGroupBox:
        box = QGroupBox("Flags", parent)
        lay = QHBoxLayout(box)
        lay.setContentsMargins(10, 10, 10, 10)
        lay.setSpacing(10)

        self.chk_res = QCheckBox("Residuals", box)
        self.chk_bn = QCheckBox("Batch norm", box)
        self.chk_vsn = QCheckBox("VSN", box)

        self.binder.bind_checkbox("use_residuals", self.chk_res)
        self.binder.bind_checkbox("use_batch_norm", self.chk_bn)
        self.binder.bind_checkbox("use_vsn", self.chk_vsn)

        lay.addWidget(self.chk_res, 0)
        lay.addWidget(self.chk_bn, 0)
        lay.addWidget(self.chk_vsn, 0)
        lay.addStretch(1)

        return box

    def _build_advanced_box(self, parent: QWidget) -> QGroupBox:
        box = QGroupBox("Advanced", parent)
        outer = QVBoxLayout(box)
        outer.setContentsMargins(10, 10, 10, 10)
        outer.setSpacing(6)

        self.exp = _Expander("Advanced knobs", parent=box)
        outer.addWidget(self.exp, 0)

        self.sp_mem = QSpinBox(box)
        self.sp_mem.setRange(1, 512)
        self.binder.bind_spin_box("memory_size", self.sp_mem)

        self.sp_vsn = QSpinBox(box)
        self.sp_vsn.setRange(4, 512)
        self.binder.bind_spin_box("vsn_units", self.sp_vsn)

        self.exp.body_l.addWidget(QLabel("Memory:", box), 0, 0)
        self.exp.body_l.addWidget(self.sp_mem, 0, 1)
        self.exp.body_l.addWidget(QLabel("VSN units:", box), 1, 0)
        self.exp.body_l.addWidget(self.sp_vsn, 1, 1)

        return box

    def _build_preview_box(self, parent: QWidget) -> QGroupBox:
        box = QGroupBox("Preview", parent)
        lay = QVBoxLayout(box)
        lay.setContentsMargins(10, 10, 10, 10)
        lay.setSpacing(10)

        self.lbl_preview = QLabel("", box)
        self.lbl_preview.setWordWrap(True)

        self.btn_cfg = QPushButton("Architecture config…", box)
        self.btn_cfg.setCursor(Qt.PointingHandCursor)
        self.btn_cfg.clicked.connect(self._open_arch_dialog)

        lay.addWidget(self.lbl_preview, 0)
        lay.addWidget(self.btn_cfg, 0)

        return box

    # -----------------------------------------------------------------
    # Wiring
    # -----------------------------------------------------------------
    def _wire(self) -> None:
        self.cmb_model.currentIndexChanged.connect(
            self._on_model_selected,
        )

        self.chk_vsn.toggled.connect(self._toggle_vsn)
        self._toggle_vsn(self.chk_vsn.isChecked())

        self.store.config_changed.connect(
            lambda _k: self._sync_preview(),
        )
        self.store.config_replaced.connect(
            lambda _cfg: self._sync_preview(),
        )

    def _toggle_vsn(self, on: bool) -> None:
        self.sp_vsn.setEnabled(bool(on))

    # -----------------------------------------------------------------
    # Model selection logic
    # -----------------------------------------------------------------
    def _on_model_selected(self) -> None:
        model = self.cmb_model.currentData()
        model = str(model or "").strip()

        if model == "PoroElasticSubsNet":
            msg = (
                "⏳ PoroElasticSubsNet is planned for the "
                "next release.\n\n"
                "For now we keep GeoPriorSubsNet and set "
                "PDE mode to 'consolidation' to approximate "
                "consolidation-only physics."
            )
            self._fallback_model(
                msg=msg,
                pde_mode="consolidation",
            )
            return

        if model == "HybridAttn":
            msg = (
                "🧪 HybridAttn is a roadmap item.\n\n"
                "To mimic a pure Hybrid attention run, we "
                "keep GeoPriorSubsNet and disable physics "
                "(PDE mode = 'off').\n\n"
                "HybridAttn + anomaly detection will land "
                "in a future release."
            )
            self._fallback_model(msg=msg, pde_mode="off")
            return

        self.lbl_banner.setVisible(False)

    def _fallback_model(
        self,
        *,
        msg: str,
        pde_mode: str,
    ) -> None:
        self.lbl_banner.setText(str(msg))
        self.lbl_banner.setVisible(True)

        idx = self.cmb_model.findData("GeoPriorSubsNet")
        if idx >= 0:
            with QSignalBlocker(self.cmb_model):
                self.cmb_model.setCurrentIndex(idx)

        with self.store.batch():
            self.store.patch({"model_name": "GeoPriorSubsNet"})
            self.store.patch({"pde_mode": str(pde_mode)})

    # -----------------------------------------------------------------
    # Architecture dialog
    # -----------------------------------------------------------------
    def _open_arch_dialog(self) -> None:
        cfg = self.store.cfg
        base_cfg = getattr(cfg, "_base_cfg", {}) or {}

        cur = self._seed_dialog_payload()

        dlg = ArchitectureConfigDialog(
            base_cfg=base_cfg,
            current_overrides=cur,
            parent=self,
        )
        if dlg.exec_() != dlg.Accepted:
            return

        overrides = dlg.get_overrides()
        patch = self._map_dialog_overrides(overrides)
        if patch:
            self.store.patch(patch)

    def _seed_dialog_payload(self) -> Dict[str, Any]:
        cfg = self.store.cfg
        return {
            "ATTENTION_LEVELS": list(cfg.attention_levels),
            "EMBED_DIM": int(cfg.embed_dim),
            "HIDDEN_UNITS": int(cfg.hidden_units),
            "LSTM_UNITS": int(cfg.lstm_units),
            "ATTENTION_UNITS": int(cfg.attention_units),
            "NUMBER_HEADS": int(cfg.num_heads),
            "DROPOUT_RATE": float(cfg.dropout_rate),
            "MEMORY_SIZE": int(cfg.memory_size),
            "SCALES": list(cfg.scales),
            "USE_RESIDUALS": bool(cfg.use_residuals),
            "USE_BATCH_NORM": bool(cfg.use_batch_norm),
            "USE_VSN": bool(cfg.use_vsn),
            "VSN_UNITS": int(cfg.vsn_units),
        }

    def _map_dialog_overrides(
        self,
        overrides: Dict[str, Any],
    ) -> Dict[str, Any]:
        mapping = {
            "ATTENTION_LEVELS": "attention_levels",
            "EMBED_DIM": "embed_dim",
            "HIDDEN_UNITS": "hidden_units",
            "LSTM_UNITS": "lstm_units",
            "ATTENTION_UNITS": "attention_units",
            "NUMBER_HEADS": "num_heads",
            "DROPOUT_RATE": "dropout_rate",
            "MEMORY_SIZE": "memory_size",
            "SCALES": "scales",
            "USE_RESIDUALS": "use_residuals",
            "USE_BATCH_NORM": "use_batch_norm",
            "USE_VSN": "use_vsn",
            "VSN_UNITS": "vsn_units",
        }

        patch: Dict[str, Any] = {}
        for k, v in (overrides or {}).items():
            key = mapping.get(str(k))
            if key is not None:
                patch[key] = v

        return patch

    # -----------------------------------------------------------------
    # Preview
    # -----------------------------------------------------------------
    def _sync_preview(self) -> None:
        cfg = self.store.cfg

        flags = []
        if cfg.use_residuals:
            flags.append("residuals")
        if cfg.use_batch_norm:
            flags.append("batch-norm")
        if cfg.use_vsn:
            flags.append("vsn")

        flag_txt = ", ".join(flags) if flags else "none"

        txt = (
            f"Model: {cfg.model_name}\n"
            f"PDE mode: {cfg.pde_mode}\n\n"
            f"embed={cfg.embed_dim}, hidden={cfg.hidden_units}, "
            f"lstm={cfg.lstm_units}\n"
            f"att={cfg.attention_units}, heads={cfg.num_heads}, "
            f"dropout={cfg.dropout_rate:.3f}\n"
            f"memory={cfg.memory_size}, vsn_units={cfg.vsn_units}\n"
            f"flags: {flag_txt}\n"
            f"attention: {', '.join(cfg.attention_levels)}"
        )

        self.lbl_preview.setText(txt)

        custom_keys = (
            "model_name",
            "embed_dim",
            "hidden_units",
            "lstm_units",
            "attention_units",
            "num_heads",
            "dropout_rate",
            "memory_size",
            "scales",
            "use_residuals",
            "use_batch_norm",
            "use_vsn",
            "vsn_units",
            "attention_levels",
        )

        custom = any(
            self.store.is_overridden(k)
            for k in custom_keys
        )

        self.badge(
            "arch",
            text="Custom" if custom else "Default",
            accent="warn" if custom else "ok",
            tip="Architecture override status",
        )
