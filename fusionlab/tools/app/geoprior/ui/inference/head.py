# geoprior/ui/inference/head.py
# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>

from __future__ import annotations

from typing import Optional, Sequence

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QIcon,  QFontMetrics
from PyQt5.QtWidgets import (
    QHBoxLayout,
    QToolButton,
    QWidget,
    QLabel, 
    QSizePolicy, 
    QStyle, 
    QLineEdit
)

from ...config.store import GeoConfigStore
from ...config.prior_schema import FieldKey 
from ..icon_utils import try_icon

__all__ = ["InferenceHeadBar"]


class _Chip(QLabel):
    def __init__(self, text: str, *, kind: str, parent=None):
        super().__init__(text, parent)
        self.setObjectName("inferChip")
        self.setProperty("kind", kind)
        self.setAlignment(Qt.AlignCenter)
        self.setSizePolicy(
            QSizePolicy.Minimum,
            QSizePolicy.Fixed,
        )

class _ElideLabel(QLabel):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self._full = ""

    def set_full_text(self, text: str) -> None:
        self._full = str(text or "")
        self._apply_elide()

    def resizeEvent(self, e) -> None:
        super().resizeEvent(e)
        self._apply_elide()

    def _apply_elide(self) -> None:
        fm = QFontMetrics(self.font())
        w = max(10, self.width())
        txt = fm.elidedText(self._full, Qt.ElideRight, w)
        super().setText(txt)

class InferenceHeadBar(QWidget):
    """
    Head [B]: compact global bar (right work area only).

    - Plan summary is computed from store (plan.py).
    - Runtime strip (Source/Future) is set by the tab.
    - Uses Train/Tune styling by reusing:
      QWidget#trainTopBar, QLabel#sumLine, QToolButton#miniAction
    """

    help_clicked = pyqtSignal()

    def __init__(
        self,
        *,
        store: GeoConfigStore,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)

        self._store = store

        self._source = "val"
        self._future = False

        self._build_ui()
        self._wire()

        self.refresh_from_store()

    # ---------------------------------------------------------
    # Public API
    # ---------------------------------------------------------
    def refresh_from_store(self) -> None:
        s = self._store
        cov = float(s.get_value(FieldKey("interval_level"), default=0.80))
        mode = str(s.get_value(FieldKey("calibration_mode"), default="none"))
        tmp = float(s.get_value(FieldKey("calibration_temperature"), default=1.0))
    
        inc = bool(s.get("infer.include_gwl", False))
        plt = bool(s.get("infer.plots", True))
    
        self.lbl_plan.set_full_text("Uncertainty & calibration settings")
    
        self._set_chip(self.chip_cov, f"{cov*100:.0f}%", "info")
        self._set_chip(self.chip_cal, f"Calib: {mode}", "off" if mode=="none" else "ok")
        self._set_chip(self.chip_tmp, f"T: {tmp:.1f}", "info")
        self._set_chip(self.chip_gwl, "GWL: on" if inc else "GWL: off",
                       "ok" if inc else "off")
        self._set_chip(self.chip_plot, "Plots: on" if plt else "Plots: off",
                       "ok" if plt else "off")
    
        self._fit_chips()
    
    def _set_chip(self, chip: QLabel, txt: str, kind: str) -> None:
        chip.setText(txt)
        chip.setProperty("kind", kind)
        chip.style().unpolish(chip)
        chip.style().polish(chip)


    def set_runtime_mode(
        self,
        *,
        source: str,
        future: bool,
    ) -> None:
        self._source = str(source or "").strip() or "—"
        self._future = bool(future)
        self._sync_mode_strip()

    # ---------------------------------------------------------
    # UI
    # ---------------------------------------------------------
    def _build_ui(self) -> None:
        # Reuse Train top bar styling
        self.setObjectName("trainTopBar")
    
        root = QHBoxLayout(self)
        root.setContentsMargins(8, 4, 8, 4)
        root.setSpacing(8)
    
        # Title (fixed)
        self.lbl_title = QLabel("Inference", self)
        self.lbl_title.setObjectName("inferHeadTitle")
        self.lbl_title.setSizePolicy(
            QSizePolicy.Fixed,
            QSizePolicy.Fixed,
        )
    
        # Elided plan line (takes the stretch)
        self.lbl_plan = _ElideLabel("", self)
        self.lbl_plan.setObjectName("sumLine")
        self.lbl_plan.setWordWrap(False)
        self.lbl_plan.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Fixed,
        )
    
        # Chips (shown/hidden by _fit_chips)
        self.chip_cov = _Chip("80%", kind="info", parent=self)
        self.chip_cal = _Chip("Calib: none", kind="off", parent=self)
        self.chip_tmp = _Chip("T: 1.0", kind="info", parent=self)
        self.chip_gwl = _Chip("GWL: off", kind="off", parent=self)
        self.chip_plot = _Chip("Plots: on", kind="ok", parent=self)
    
        self._chips = [
            self.chip_cov,
            self.chip_cal,
            self.chip_tmp,
            self.chip_gwl,
            self.chip_plot,
        ]
    
        # Runtime strip (Source/Future)
        self.lbl_mode = QLabel("", self)
        self.lbl_mode.setObjectName("sumLine")
        self.lbl_mode.setWordWrap(False)
        self.lbl_mode.setSizePolicy(
            QSizePolicy.Minimum,
            QSizePolicy.Fixed,
        )
        self._sync_mode_strip()
    
        # Filter button
        self.btn_filter = self._mk_icon_btn(
            icon_names=("filter2.svg", "filter.svg", "search.svg"),
            fallback_text="🔎",
            tip="Filter / search",
            fallback_std=QStyle.SP_FileDialogContentsView,
        )
        self.btn_filter.setObjectName("miniAction")
    
        # Search field
        self.ed_search = QLineEdit(self)
        self.ed_search.setObjectName("headSearch")
        self.ed_search.setPlaceholderText("Search settings...")
        self.ed_search.setClearButtonEnabled(True)
        self.ed_search.setMaximumWidth(260)
        self.ed_search.setSizePolicy(
            QSizePolicy.Fixed,
            QSizePolicy.Fixed,
        )
        self.btn_help = self._mk_icon_btn(
            icon_names=("help.svg", "question.svg", "info.svg"),
            fallback_text="?",
            tip="Help",
        )
    
        # ---------------- Layout ----------------
        root.addWidget(self.lbl_title, 0)
        root.addWidget(self.lbl_plan, 1)
    
        # chips sit right after plan line
        for c in self._chips:
            root.addWidget(c, 0)
    
        # mode + filter/search + help on the far right
        root.addWidget(self.lbl_mode, 0)
        root.addSpacing(4)
        root.addWidget(self.btn_filter, 0)
        root.addWidget(self.ed_search, 0)
        root.addWidget(self.btn_help, 0)
    
        # initial fit
        self._fit_chips()

        
    def _fit_chips(self) -> None:
        for c in self._chips:
            c.show()
    
        # keep these always if possible
        must_keep = {self.chip_cov, self.chip_cal}
    
        # available width budget (rough)
        budget = self.width() - 220  # actions area
        used = self.lbl_title.sizeHint().width()
        used += self.lbl_mode.sizeHint().width()
    
        # hide from the end first
        for c in self._chips:
            used += c.sizeHint().width() + 6
    
        if used <= budget:
            return
    
        for c in reversed(self._chips):
            if c in must_keep:
                continue
            c.hide()
            used -= c.sizeHint().width() + 6
            if used <= budget:
                break
    
    def resizeEvent(self, e) -> None:
        super().resizeEvent(e)
        self._fit_chips()

    def _mk_icon_btn(
        self,
        *,
        icon_names: Sequence[str],
        fallback_text: str,
        tip: str,
        fallback_std: Optional[QStyle.StandardPixmap] = None,
    ) -> QToolButton:
        b = QToolButton(self)
        b.setObjectName("miniAction")
        b.setToolTip(tip)
        b.setAutoRaise(True)
        b.setCursor(Qt.PointingHandCursor)
        b.setToolButtonStyle(Qt.ToolButtonIconOnly)
        b.setFixedSize(28, 28)
    
        ico: Optional[QIcon] = None
        for nm in icon_names:
            ico = try_icon(nm)
            if ico is not None:
                break
    
        if ico is None and fallback_std is not None:
            ico = self.style().standardIcon(fallback_std)
    
        if ico is not None:
            b.setIcon(ico)
        else:
            b.setToolButtonStyle(Qt.ToolButtonTextOnly)
            b.setText(fallback_text)
    
        return b


    def _sync_mode_strip(self) -> None:
        fut = "ON" if self._future else "OFF"
        self.lbl_mode.setText(
            f"Source: {self._source}  •  Future: {fut}"
        )

    # ---------------------------------------------------------
    # Wiring
    # ---------------------------------------------------------
    def _wire(self) -> None:
        self.btn_help.clicked.connect(
            self.help_clicked.emit
        )

        # Store -> head summary
        self._store.config_changed.connect(
            lambda _k: self.refresh_from_store()
        )
