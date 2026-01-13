# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>

"""
Dialog for configuring probabilistic outputs and pinball
loss weights (store-driven, v3.2):

- quantiles
- subs_weights
- gwl_weights

Uncertainty controls:
- interval_level
- crossing_penalty
- crossing_margin
- calibration_mode
- calibration_temperature
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFrame,
    QGridLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QVBoxLayout,
)

from ..config.store import GeoConfigStore
from ..config.prior_schema import FieldKey


class ProbConfigDialog(QDialog):
    """
    Edit probabilistic settings via GeoConfigStore.

    Notes
    -----
    - Writes are applied only on OK.
    - Cancel does not touch the store.
    """

    def __init__(
        self,
        *,
        store: GeoConfigStore,
        parent: Optional[QDialog] = None,
    ) -> None:
        super().__init__(parent)

        self._store = store
        self.setWindowTitle("Probabilistic configuration")

        # ----------------------------
        # Load initial store values
        # ----------------------------
        q0 = self._coerce_quantiles(
            self._get("quantiles", [0.1, 0.5, 0.9]),
        )

        subs0 = self._coerce_weights(
            self._get("subs_weights", {}),
            q0,
            task="subs",
        )
        gwl0 = self._coerce_weights(
            self._get("gwl_weights", {}),
            q0,
            task="gwl",
        )

        interval0 = self._coerce_float(
            self._get("interval_level", 0.8),
            default=0.8,
        )
        cross_pen0 = self._coerce_float(
            self._get("crossing_penalty", 0.0),
            default=0.0,
        )
        cross_m0 = self._coerce_float(
            self._get("crossing_margin", 0.0),
            default=0.0,
        )
        cal_mode0 = str(self._get("calibration_mode", "none") or "none")
        cal_temp0 = self._coerce_float(
            self._get("calibration_temperature", 1.0),
            default=1.0,
        )

        self._initial_q = list(q0)
        self._initial_subs = dict(subs0)
        self._initial_gwl = dict(gwl0)

        self._initial_interval = float(interval0)
        self._initial_cross_pen = float(cross_pen0)
        self._initial_cross_margin = float(cross_m0)
        self._initial_cal_mode = str(cal_mode0)
        self._initial_cal_temp = float(cal_temp0)

        # ----------------------------
        # Layout
        # ----------------------------
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(12)

        # ==================================================
        # Card 1: Quantiles + weights
        # ==================================================
        card1, grid1 = self._make_card(
            "Pinball loss (quantiles and weights)",
        )

        self.le_quantiles = QLineEdit()
        self.le_quantiles.setPlaceholderText("e.g. 0.1, 0.5, 0.9")
        self.le_quantiles.setText(
            ", ".join(self._fmt_float(x) for x in q0),
        )

        self.le_subs = QLineEdit()
        self.le_subs.setPlaceholderText(
            "e.g. 0.1:3.0, 0.5:1.0, 0.9:3.0",
        )
        self.le_subs.setText(self._weights_to_str(q0, subs0))

        self.le_gwl = QLineEdit()
        self.le_gwl.setPlaceholderText(
            "e.g. 0.1:1.5, 0.5:1.0, 0.9:1.5",
        )
        self.le_gwl.setText(self._weights_to_str(q0, gwl0))

        r = 0
        grid1.addWidget(self._lbl("Quantiles:"), r, 0)
        grid1.addWidget(self.le_quantiles, r, 1)
        r += 1

        grid1.addWidget(self._lbl("Subsidence weights:"), r, 0)
        grid1.addWidget(self.le_subs, r, 1)
        r += 1

        grid1.addWidget(self._lbl("GWL weights:"), r, 0)
        grid1.addWidget(self.le_gwl, r, 1)

        layout.addWidget(card1)

        # ==================================================
        # Card 2: Uncertainty controls
        # ==================================================
        card2, grid2 = self._make_card(
            "Uncertainty & calibration",
        )

        self.sp_interval = QDoubleSpinBox()
        self.sp_interval.setDecimals(3)
        self.sp_interval.setRange(0.001, 0.999)
        self.sp_interval.setSingleStep(0.05)
        self.sp_interval.setValue(float(interval0))
        self.sp_interval.setToolTip(
            "Interval level in (0, 1), e.g. 0.80 for 80%.",
        )

        self.sp_cross_pen = QDoubleSpinBox()
        self.sp_cross_pen.setDecimals(6)
        self.sp_cross_pen.setRange(0.0, 1e9)
        self.sp_cross_pen.setSingleStep(0.1)
        self.sp_cross_pen.setValue(float(cross_pen0))
        self.sp_cross_pen.setToolTip(
            "Penalty strength for quantile crossing.",
        )

        self.sp_cross_margin = QDoubleSpinBox()
        self.sp_cross_margin.setDecimals(6)
        self.sp_cross_margin.setRange(0.0, 1e9)
        self.sp_cross_margin.setSingleStep(0.01)
        self.sp_cross_margin.setValue(float(cross_m0))
        self.sp_cross_margin.setToolTip(
            "Optional margin for crossing constraint.",
        )

        self.cb_cal_mode = QComboBox()
        self.cb_cal_mode.addItems(
            ["none", "temperature", "isotonic", "conformal"],
        )
        self.cb_cal_mode.setCurrentText(
            cal_mode0 if cal_mode0 else "none",
        )
        self.cb_cal_mode.currentTextChanged.connect(
            self._on_cal_mode_changed,
        )

        self.sp_cal_temp = QDoubleSpinBox()
        self.sp_cal_temp.setDecimals(6)
        self.sp_cal_temp.setRange(1e-6, 1e6)
        self.sp_cal_temp.setSingleStep(0.1)
        self.sp_cal_temp.setValue(float(cal_temp0))
        self.sp_cal_temp.setToolTip(
            "Temperature scaling parameter (> 0).",
        )

        rr = 0
        grid2.addWidget(self._lbl("Interval level:"), rr, 0)
        grid2.addWidget(self.sp_interval, rr, 1)
        rr += 1

        grid2.addWidget(self._lbl("Crossing penalty:"), rr, 0)
        grid2.addWidget(self.sp_cross_pen, rr, 1)
        rr += 1

        grid2.addWidget(self._lbl("Crossing margin:"), rr, 0)
        grid2.addWidget(self.sp_cross_margin, rr, 1)
        rr += 1

        grid2.addWidget(self._lbl("Calibration mode:"), rr, 0)
        grid2.addWidget(self.cb_cal_mode, rr, 1)
        rr += 1

        grid2.addWidget(self._lbl("Calibration temperature:"), rr, 0)
        grid2.addWidget(self.sp_cal_temp, rr, 1)

        layout.addWidget(card2)

        self._on_cal_mode_changed(self.cb_cal_mode.currentText())

        # ----------------------------
        # Buttons
        # ----------------------------
        btns = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel,
        )
        btns.accepted.connect(self._on_accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

        ok_btn = btns.button(QDialogButtonBox.Ok)
        if ok_btn is not None:
            ok_btn.setDefault(True)

    # -----------------------------------------------------------------
    # UI helpers
    # -----------------------------------------------------------------
    def _make_card(
        self,
        title: str,
    ) -> tuple[QFrame, QGridLayout]:
        frame = QFrame()
        frame.setObjectName("card")

        outer = QVBoxLayout(frame)
        outer.setContentsMargins(12, 10, 12, 12)
        outer.setSpacing(8)

        ttl = QLabel(title)
        ttl.setObjectName("cardTitle")
        outer.addWidget(ttl)

        grid = QGridLayout()
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(8)
        grid.setColumnStretch(1, 1)
        outer.addLayout(grid)

        return frame, grid

    def _lbl(self, text: str) -> QLabel:
        lab = QLabel(text)
        lab.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        return lab

    def _on_cal_mode_changed(self, mode: str) -> None:
        mode = str(mode or "").strip().lower()
        self.sp_cal_temp.setEnabled(mode == "temperature")

    # -----------------------------------------------------------------
    # Public entry
    # -----------------------------------------------------------------
    @classmethod
    def edit(
        cls,
        *,
        store: GeoConfigStore,
        parent: Optional[QDialog] = None,
    ) -> bool:
        dlg = cls(store=store, parent=parent)
        return dlg.exec_() == QDialog.Accepted

    # -----------------------------------------------------------------
    # Store helpers
    # -----------------------------------------------------------------
    def _get(self, key: str, default: Any) -> Any:
        try:
            return self._store.get_value(FieldKey(key))
        except Exception:
            return default

    def _patch(self, patch: Dict[str, Any]) -> None:
        with self._store.batch():
            self._store.patch(dict(patch))

    # -----------------------------------------------------------------
    # Parsing / formatting
    # -----------------------------------------------------------------
    def _coerce_float(
        self,
        obj: Any,
        *,
        default: float,
    ) -> float:
        try:
            return float(obj)
        except Exception:
            return float(default)

    def _float_diff(self, a: float, b: float) -> bool:
        return abs(float(a) - float(b)) > 1e-12

    def _fmt_float(self, x: float) -> str:
        s = f"{float(x):.6f}"
        s = s.rstrip("0").rstrip(".")
        return s or "0"

    def _parse_float_list(self, text: str) -> List[float]:
        text = (text or "").strip()
        if not text:
            return []
        out: List[float] = []
        for part in text.split(","):
            s = part.strip()
            if not s:
                continue
            out.append(float(s))
        return out

    def _coerce_quantiles(self, obj: Any) -> List[float]:
        qs: List[float] = []
        if isinstance(obj, (list, tuple)):
            for x in obj:
                try:
                    qs.append(float(x))
                except Exception:
                    continue
        if not qs:
            qs = [0.1, 0.5, 0.9]

        uniq = sorted({round(float(q), 6) for q in qs})
        for q in uniq:
            if q <= 0.0 or q >= 1.0:
                raise ValueError(
                    "Quantiles must be in (0, 1).",
                )
        return [float(q) for q in uniq]

    def _default_weights(
        self,
        qs: List[float],
        *,
        task: str,
    ) -> Dict[float, float]:
        out: Dict[float, float] = {}
        for q in qs:
            if abs(q - 0.5) < 1e-6:
                out[q] = 1.0
            else:
                out[q] = 3.0 if task == "subs" else 1.5
        return out

    def _coerce_weights(
        self,
        obj: Any,
        qs: List[float],
        *,
        task: str,
    ) -> Dict[float, float]:
        base = self._default_weights(qs, task=task)
        if not isinstance(obj, dict):
            return dict(base)

        out: Dict[float, float] = {}
        for k, v in obj.items():
            try:
                q = float(k)
                out[q] = float(v)
            except Exception:
                continue

        for q in qs:
            if q not in out:
                out[q] = base[q]

        return out

    def _weights_to_str(
        self,
        qs: List[float],
        w: Dict[float, float],
    ) -> str:
        parts: List[str] = []
        for q in qs:
            if q not in w:
                continue
            parts.append(
                f"{self._fmt_float(q)}:"
                f"{self._fmt_float(w[q])}",
            )
        return ", ".join(parts)

    def _parse_weights(
        self,
        text: str,
        qs: List[float],
    ) -> Dict[float, float]:
        text = (text or "").strip()
        if not text:
            return {}

        allowed = {round(float(q), 6) for q in qs}
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

            if round(q, 6) not in allowed:
                raise ValueError(
                    f"Quantile {q} not in quantiles.",
                )
            if w < 0.0:
                raise ValueError(
                    "Weights must be >= 0.",
                )
            out[q] = w

        return out

    # -----------------------------------------------------------------
    # OK / apply
    # -----------------------------------------------------------------
    def _on_accept(self) -> None:
        try:
            qs = self._parse_float_list(self.le_quantiles.text())
            qs = self._coerce_quantiles(qs)
        except Exception as exc:
            QMessageBox.warning(
                self,
                "Invalid quantiles",
                str(exc) or "Use: 0.1, 0.5, 0.9",
            )
            return

        try:
            subs_in = self._parse_weights(self.le_subs.text(), qs)
            gwl_in = self._parse_weights(self.le_gwl.text(), qs)
        except Exception as exc:
            QMessageBox.warning(
                self,
                "Invalid weights",
                str(exc),
            )
            return

        interval = float(self.sp_interval.value())
        if interval <= 0.0 or interval >= 1.0:
            QMessageBox.warning(
                self,
                "Invalid interval level",
                "Interval level must be in (0, 1).",
            )
            return

        cross_pen = float(self.sp_cross_pen.value())
        cross_m = float(self.sp_cross_margin.value())

        cal_mode = str(self.cb_cal_mode.currentText() or "none")
        cal_mode = cal_mode.strip().lower()

        cal_temp = float(self.sp_cal_temp.value())
        if cal_mode == "temperature" and cal_temp <= 0.0:
            QMessageBox.warning(
                self,
                "Invalid temperature",
                "Temperature must be > 0.",
            )
            return

        subs = self._default_weights(qs, task="subs")
        gwl = self._default_weights(qs, task="gwl")

        subs.update(subs_in)
        gwl.update(gwl_in)

        patch: Dict[str, Any] = {}

        if qs != self._initial_q:
            patch["quantiles"] = qs
        if subs != self._initial_subs:
            patch["subs_weights"] = subs
        if gwl != self._initial_gwl:
            patch["gwl_weights"] = gwl

        if self._float_diff(interval, self._initial_interval):
            patch["interval_level"] = interval
        if self._float_diff(cross_pen, self._initial_cross_pen):
            patch["crossing_penalty"] = cross_pen
        if self._float_diff(cross_m, self._initial_cross_margin):
            patch["crossing_margin"] = cross_m

        if cal_mode != self._initial_cal_mode:
            patch["calibration_mode"] = cal_mode
        if self._float_diff(cal_temp, self._initial_cal_temp):
            patch["calibration_temperature"] = cal_temp

        if patch:
            self._patch(patch)

        self.accept()

    @staticmethod
    def snapshot(store: GeoConfigStore) -> Dict[str, Any]:
        keys = (
            "quantiles",
            "subs_weights",
            "gwl_weights",
            "interval_level",
            "crossing_penalty",
            "crossing_margin",
            "calibration_mode",
            "calibration_temperature",
        )
        snap: Dict[str, Any] = {}
        for k in keys:
            try:
                snap[k] = store.get_value(FieldKey(k))
            except Exception:
                snap[k] = None
        return snap
