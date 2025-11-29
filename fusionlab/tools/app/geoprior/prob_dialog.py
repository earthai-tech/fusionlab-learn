# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>

"""
Dialog for configuring probabilistic outputs and pinball
loss weights:

- QUANTILES
- SUBS_WEIGHTS
- GWL_WEIGHTS
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from PyQt5.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QGridLayout,
    QLabel,
    QLineEdit,
    QDialogButtonBox,
    QMessageBox,
)


class ProbConfigDialog(QDialog):
    """Dialog to edit quantiles and loss weights."""

    def __init__(
        self,
        *,
        base_cfg: Dict[str, Any],
        current_overrides: Optional[Dict[str, Any]] = None,
        parent: Optional["QDialog"] = None,
    ) -> None:
        super().__init__(parent)

        self.setWindowTitle("Probabilistic config")

        cfg: Dict[str, Any] = dict(base_cfg or {})
        if current_overrides:
            cfg.update(current_overrides)

        def get(key: str, default: Any) -> Any:
            return cfg.get(key, default)

        quantiles = get("QUANTILES", [0.1, 0.5, 0.9]) or []
        try:
            q_list = [float(q) for q in quantiles]
        except Exception:
            q_list = [0.1, 0.5, 0.9]

        def coerce_weights(
            obj: Any,
            default: Dict[float, float],
        ) -> Dict[float, float]:
            if not isinstance(obj, dict):
                return dict(default)
            w: Dict[float, float] = {}
            for k, v in obj.items():
                try:
                    q = float(k)
                    w[q] = float(v)
                except Exception:
                    continue
            if not w:
                return dict(default)
            return w

        subs_default = {0.1: 3.0, 0.5: 1.0, 0.9: 3.0}
        gwl_default = {0.1: 1.5, 0.5: 1.0, 0.9: 1.5}

        subs_w = coerce_weights(
            get("SUBS_WEIGHTS", subs_default),
            subs_default,
        )
        gwl_w = coerce_weights(
            get("GWL_WEIGHTS", gwl_default),
            gwl_default,
        )

        self._initial: Dict[str, Any] = {
            "QUANTILES": list(q_list),
            "SUBS_WEIGHTS": dict(subs_w),
            "GWL_WEIGHTS": dict(gwl_w),
        }
        self._overrides: Dict[str, Any] = {}

        layout = QVBoxLayout(self)
        grid = QGridLayout()
        row = 0

        self.le_quantiles = QLineEdit()
        self.le_quantiles.setPlaceholderText("e.g. 0.1, 0.5, 0.9")
        self.le_quantiles.setText(
            ", ".join(self._fmt_float(q) for q in q_list)
        )

        grid.addWidget(QLabel("Quantiles:"), row, 0)
        grid.addWidget(self.le_quantiles, row, 1)
        row += 1

        self.le_subs = QLineEdit()
        self.le_subs.setPlaceholderText(
            "e.g. 0.1:3.0, 0.5:1.0, 0.9:3.0"
        )
        self.le_subs.setText(
            self._weights_to_str(q_list, subs_w)
        )

        grid.addWidget(
            QLabel("Subsidence weights:"),
            row,
            0,
        )
        grid.addWidget(self.le_subs, row, 1)
        row += 1

        self.le_gwl = QLineEdit()
        self.le_gwl.setPlaceholderText(
            "e.g. 0.1:1.5, 0.5:1.0, 0.9:1.5"
        )
        self.le_gwl.setText(
            self._weights_to_str(q_list, gwl_w)
        )

        grid.addWidget(
            QLabel("GWL weights:"),
            row,
            0,
        )
        grid.addWidget(self.le_gwl, row, 1)
        row += 1

        layout.addLayout(grid)

        btn_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        btn_box.accepted.connect(self._on_accept)
        btn_box.rejected.connect(self.reject)
        layout.addWidget(btn_box)

    # ------------------------------------------------------------------
    def _fmt_float(self, x: float) -> str:
        s = f"{x:.4f}"
        s = s.rstrip("0").rstrip(".")
        return s or "0"

    def _weights_to_str(
        self,
        quantiles: List[float],
        weights: Dict[float, float],
    ) -> str:
        parts: List[str] = []
        for q in quantiles:
            w = weights.get(q)
            if w is None:
                continue
            parts.append(
                f"{self._fmt_float(q)}:"
                f"{self._fmt_float(float(w))}"
            )
        return ", ".join(parts)

    def _parse_float_list(
        self,
        text: str,
        fallback: List[float],
    ) -> List[float]:
        text = text.strip()
        if not text:
            return list(fallback)
        vals: List[float] = []
        for part in text.split(","):
            s = part.strip()
            if not s:
                continue
            vals.append(float(s))
        if not vals:
            return list(fallback)
        return vals

    def _parse_weights(
        self,
        text: str,
        quantiles: List[float],
    ) -> Dict[float, float]:
        text = text.strip()
        if not text:
            return {}
        allowed = {round(q, 6) for q in quantiles}
        out: Dict[float, float] = {}
        for part in text.split(","):
            s = part.strip()
            if not s:
                continue
            if ":" not in s:
                raise ValueError("Missing ':' in weight pair.")
            q_str, w_str = s.split(":", 1)
            q = float(q_str.strip())
            w = float(w_str.strip())
            rq = round(q, 6)
            if rq not in allowed:
                raise ValueError(
                    f"Quantile {q} not in QUANTILES."
                )
            out[q] = w
        return out

    def _on_accept(self) -> None:
        try:
            q_list = self._parse_float_list(
                self.le_quantiles.text(),
                self._initial["QUANTILES"],
            )
        except Exception:
            QMessageBox.warning(
                self,
                "Invalid quantiles",
                "Quantiles must be a comma-separated list "
                "of floats, e.g. '0.1, 0.5, 0.9'.",
            )
            return

        try:
            subs_w = self._parse_weights(
                self.le_subs.text(),
                q_list,
            )
        except Exception as exc:
            QMessageBox.warning(
                self,
                "Invalid subsidence weights",
                str(exc),
            )
            return

        try:
            gwl_w = self._parse_weights(
                self.le_gwl.text(),
                q_list,
            )
        except Exception as exc:
            QMessageBox.warning(
                self,
                "Invalid GWL weights",
                str(exc),
            )
            return

        current: Dict[str, Any] = {
            "QUANTILES": q_list,
            "SUBS_WEIGHTS": subs_w or self._initial["SUBS_WEIGHTS"],
            "GWL_WEIGHTS": gwl_w or self._initial["GWL_WEIGHTS"],
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
