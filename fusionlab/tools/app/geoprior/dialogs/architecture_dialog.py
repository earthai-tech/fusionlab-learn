# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>

"""
Dialog for configuring attention / architecture fields used
by :class:`GeoPriorSubsNet` (BaseAttentive block).

It edits a small subset of the NATCOM config:

- ATTENTION_LEVELS
- EMBED_DIM, HIDDEN_UNITS, LSTM_UNITS, ATTENTION_UNITS
- NUMBER_HEADS, DROPOUT_RATE
- MEMORY_SIZE, SCALES
- USE_RESIDUALS, USE_BATCH_NORM, USE_VSN, VSN_UNITS
"""

from __future__ import annotations

from typing import Any, Dict, Optional, List

from PyQt5.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QGridLayout,
    QLabel,
    QSpinBox,
    QDoubleSpinBox,
    QCheckBox,
    QLineEdit,
    QDialogButtonBox,
    QMessageBox,
)


class ArchitectureConfigDialog(QDialog):
    """Small dialog to tweak BaseAttentive architecture."""

    def __init__(
        self,
        *,
        base_cfg: Dict[str, Any],
        current_overrides: Optional[Dict[str, Any]] = None,
        parent: Optional["QDialog"] = None,
    ) -> None:
        super().__init__(parent)

        self.setWindowTitle("Architecture config")

        cfg: Dict[str, Any] = dict(base_cfg or {})
        if current_overrides:
            cfg.update(current_overrides)

        def get(key: str, default: Any) -> Any:
            return cfg.get(key, default)

        levels = get(
            "ATTENTION_LEVELS",
            ["cross", "hierarchical", "memory"],
        )
        if not isinstance(levels, list):
            levels = [str(levels)]
        levels = [str(x) for x in levels]

        embed_dim = int(get("EMBED_DIM", 32))
        hidden_units = int(get("HIDDEN_UNITS", 64))
        lstm_units = int(get("LSTM_UNITS", 64))
        att_units = int(get("ATTENTION_UNITS", 32))
        n_heads = int(get("NUMBER_HEADS", 4))
        dropout = float(get("DROPOUT_RATE", 0.10))

        memory_size = int(get("MEMORY_SIZE", 50))
        scales = get("SCALES", [1, 2]) or [1, 2]
        if not isinstance(scales, list):
            scales = [scales]
        scales = [int(s) for s in scales]

        use_residuals = bool(get("USE_RESIDUALS", True))
        use_batch_norm = bool(get("USE_BATCH_NORM", False))
        use_vsn = bool(get("USE_VSN", True))
        vsn_units = int(get("VSN_UNITS", 32))

        # Store initial values so we can compute deltas
        self._initial: Dict[str, Any] = {
            "ATTENTION_LEVELS": list(levels),
            "EMBED_DIM": embed_dim,
            "HIDDEN_UNITS": hidden_units,
            "LSTM_UNITS": lstm_units,
            "ATTENTION_UNITS": att_units,
            "NUMBER_HEADS": n_heads,
            "DROPOUT_RATE": dropout,
            "MEMORY_SIZE": memory_size,
            "SCALES": list(scales),
            "USE_RESIDUALS": use_residuals,
            "USE_BATCH_NORM": use_batch_norm,
            "USE_VSN": use_vsn,
            "VSN_UNITS": vsn_units,
        }
        self._overrides: Dict[str, Any] = {}

        layout = QVBoxLayout(self)

        grid = QGridLayout()
        row = 0

        # Attention levels as three checkboxes
        grid.addWidget(QLabel("Attention levels:"), row, 0)
        self.chk_att_cross = QCheckBox("cross")
        self.chk_att_hier = QCheckBox("hierarchical")
        self.chk_att_mem = QCheckBox("memory")

        self.chk_att_cross.setChecked("cross" in levels)
        self.chk_att_hier.setChecked("hierarchical" in levels)
        self.chk_att_mem.setChecked("memory" in levels)

        grid.addWidget(self.chk_att_cross, row, 1)
        grid.addWidget(self.chk_att_hier, row, 2)
        grid.addWidget(self.chk_att_mem, row, 3)
        row += 1

        # Core dimensions
        self.sp_embed = QSpinBox()
        self.sp_embed.setRange(8, 512)
        self.sp_embed.setValue(embed_dim)

        self.sp_hidden = QSpinBox()
        self.sp_hidden.setRange(8, 1024)
        self.sp_hidden.setValue(hidden_units)

        self.sp_lstm = QSpinBox()
        self.sp_lstm.setRange(8, 1024)
        self.sp_lstm.setValue(lstm_units)

        self.sp_att = QSpinBox()
        self.sp_att.setRange(8, 512)
        self.sp_att.setValue(att_units)

        self.sp_heads = QSpinBox()
        self.sp_heads.setRange(1, 16)
        self.sp_heads.setValue(n_heads)

        self.sp_dropout = QDoubleSpinBox()
        self.sp_dropout.setDecimals(3)
        self.sp_dropout.setRange(0.0, 0.9)
        self.sp_dropout.setSingleStep(0.01)
        self.sp_dropout.setValue(dropout)

        grid.addWidget(QLabel("Embed dim:"), row, 0)
        grid.addWidget(self.sp_embed, row, 1)
        grid.addWidget(QLabel("Hidden units:"), row, 2)
        grid.addWidget(self.sp_hidden, row, 3)
        row += 1

        grid.addWidget(QLabel("LSTM units:"), row, 0)
        grid.addWidget(self.sp_lstm, row, 1)
        grid.addWidget(QLabel("Attention units:"), row, 2)
        grid.addWidget(self.sp_att, row, 3)
        row += 1

        grid.addWidget(QLabel("Num heads:"), row, 0)
        grid.addWidget(self.sp_heads, row, 1)
        grid.addWidget(QLabel("Dropout:"), row, 2)
        grid.addWidget(self.sp_dropout, row, 3)
        row += 1

        # Additional knobs
        self.sp_memory = QSpinBox()
        self.sp_memory.setRange(1, 512)
        self.sp_memory.setValue(memory_size)

        self.le_scales = QLineEdit()
        self.le_scales.setPlaceholderText("e.g. 1, 2")
        self.le_scales.setText(
            ", ".join(str(s) for s in scales)
        )

        grid.addWidget(QLabel("Memory size:"), row, 0)
        grid.addWidget(self.sp_memory, row, 1)
        grid.addWidget(QLabel("Scales:"), row, 2)
        grid.addWidget(self.le_scales, row, 3)
        row += 1

        # Booleans
        self.chk_residuals = QCheckBox("Use residuals")
        self.chk_residuals.setChecked(use_residuals)

        self.chk_batch_norm = QCheckBox("Use batch norm")
        self.chk_batch_norm.setChecked(use_batch_norm)

        self.chk_vsn = QCheckBox("Use VSN")
        self.chk_vsn.setChecked(use_vsn)

        self.sp_vsn_units = QSpinBox()
        self.sp_vsn_units.setRange(4, 512)
        self.sp_vsn_units.setValue(vsn_units)

        grid.addWidget(self.chk_residuals, row, 0)
        grid.addWidget(self.chk_batch_norm, row, 1)
        grid.addWidget(self.chk_vsn, row, 2)
        grid.addWidget(QLabel("VSN units:"), row, 3)
        grid.addWidget(self.sp_vsn_units, row, 4)
        row += 1

        layout.addLayout(grid)

        # Buttons
        btn_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        btn_box.accepted.connect(self._on_accept)
        btn_box.rejected.connect(self.reject)
        layout.addWidget(btn_box)

    # ------------------------------------------------------------------
    def _parse_int_list(
        self,
        text: str,
        fallback: List[int],
    ) -> List[int]:
        text = text.strip()
        if not text:
            return list(fallback)
        vals: List[int] = []
        for part in text.split(","):
            s = part.strip()
            if not s:
                continue
            vals.append(int(s))
        if not vals:
            return list(fallback)
        return vals

    def _on_accept(self) -> None:
        # attention levels
        levels: List[str] = []
        if self.chk_att_cross.isChecked():
            levels.append("cross")
        if self.chk_att_hier.isChecked():
            levels.append("hierarchical")
        if self.chk_att_mem.isChecked():
            levels.append("memory")

        if not levels:
            QMessageBox.warning(
                self,
                "Invalid configuration",
                "Select at least one attention level.",
            )
            return

        try:
            scales = self._parse_int_list(
                self.le_scales.text(),
                self._initial["SCALES"],
            )
        except Exception:
            QMessageBox.warning(
                self,
                "Invalid scales",
                "Scales must be a comma-separated list "
                "of integers, e.g. '1, 2'.",
            )
            return

        current: Dict[str, Any] = {
            "ATTENTION_LEVELS": levels,
            "EMBED_DIM": self.sp_embed.value(),
            "HIDDEN_UNITS": self.sp_hidden.value(),
            "LSTM_UNITS": self.sp_lstm.value(),
            "ATTENTION_UNITS": self.sp_att.value(),
            "NUMBER_HEADS": self.sp_heads.value(),
            "DROPOUT_RATE": float(self.sp_dropout.value()),
            "MEMORY_SIZE": self.sp_memory.value(),
            "SCALES": scales,
            "USE_RESIDUALS": self.chk_residuals.isChecked(),
            "USE_BATCH_NORM": self.chk_batch_norm.isChecked(),
            "USE_VSN": self.chk_vsn.isChecked(),
            "VSN_UNITS": self.sp_vsn_units.value(),
        }

        overrides: Dict[str, Any] = {}
        for k, v in current.items():
            if self._initial.get(k) != v:
                overrides[k] = v

        self._overrides = overrides
        self.accept()

    def get_overrides(self) -> Dict[str, Any]:
        """Return only the keys changed in this dialog."""
        return dict(self._overrides)
