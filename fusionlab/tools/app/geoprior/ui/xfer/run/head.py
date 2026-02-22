# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>

from __future__ import annotations

from typing import Any, Optional, Sequence

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFontMetrics, QIcon
from PyQt5.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QSizePolicy,
    QStyle,
    QToolButton,
    QWidget,
)

from ....config.store import GeoConfigStore
from ...icon_utils import try_icon

__all__ = ["XferHeadBar"]


class _Chip(QLabel):
    def __init__(
        self,
        text: str,
        *,
        kind: str,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(text, parent)
        self.setObjectName("inferChip")
        self.setProperty("kind", kind)
        self.setAlignment(Qt.AlignCenter)
        self.setSizePolicy(
            QSizePolicy.Minimum,
            QSizePolicy.Fixed,
        )


class _ElideLabel(QLabel):
    def __init__(self, *a, **k) -> None:
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


def _as_str(x: Any) -> str:
    return str(x or "").strip()


def _as_list(x: Any) -> list:
    if x is None:
        return []
    if isinstance(x, (list, tuple, set)):
        return list(x)
    return []


class XferHeadBar(QWidget):
    """
    Head [B] for Xfer (Run/Map).

    Goals
    -----
    - Same look as Inference head: trainTopBar + chips + search.
    - Prominent Run/Map segmented toggle.
    - NO run button here (run button lives in bottom bar).
    """

    mode_changed = pyqtSignal(str)
    help_clicked = pyqtSignal()
    filter_clicked = pyqtSignal()
    search_changed = pyqtSignal(str)

    def __init__(
        self,
        *,
        store: GeoConfigStore,
        overlay_key: str = "xfer.map.overlay",
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)

        self._s = store
        self._overlay_key = str(overlay_key or "").strip()

        self._mode = "run"

        self._build_ui()
        self._wire()

        self.refresh_from_store()

    # -------------------------------------------------
    # Public API
    # -------------------------------------------------
    def set_mode(self, mode: str) -> None:
        m = "map" if str(mode).lower() == "map" else "run"
        self._mode = m

        self.btn_run_mode.setChecked(m == "run")
        self.btn_map_mode.setChecked(m == "map")

        # Map-only chip
        self.chip_ov.setVisible(m == "map")

        self._fit_chips()
        self.refresh_from_store()

    def mode(self) -> str:
        return self._mode

    def refresh_from_store(self) -> None:
        s = self._s

        a = _as_str(s.get("xfer.city_a", ""))
        b = _as_str(s.get("xfer.city_b", ""))
        splits = _as_list(s.get("xfer.splits", ("val", "test")))
        cals = _as_list(
            s.get("xfer.calib_modes", ("none", "source", "target"))
        )
        strats = _as_list(s.get("xfer.strategies", None))

        last_out = _as_str(s.get("xfer.last_output", ""))
        has = bool(last_out)

        # ---- Plan line (elided) ----
        ab = "A/B set" if (a and b) else "pick cities A/B"
        ss = f"splits={len(splits) or 0}"
        cc = f"cal={len(cals) or 0}"
        st = f"strats={len(strats) or 0}"

        if self._mode == "run":
            self.lbl_plan.set_full_text(
                f"Run matrix • {ab} • {ss} • {cc} • {st}"
            )
        else:
            self.lbl_plan.set_full_text(
                f"Map view • overlays + metrics • {ab}"
            )

        # ---- Chips ----
        self._set_chip(
            self.chip_has,
            "HAS" if has else "—",
            "ok" if has else "off",
        )

        if a and b:
            self._set_chip(self.chip_ab, "A/B", "ok")
        elif a or b:
            self._set_chip(self.chip_ab, "A/B?", "warn")
        else:
            self._set_chip(self.chip_ab, "A/B", "off")

        if len(splits) > 0:
            self._set_chip(self.chip_sp, f"S:{len(splits)}", "ok")
        else:
            self._set_chip(self.chip_sp, "S:0", "warn")

        if len(cals) > 0:
            self._set_chip(self.chip_cal, f"Cal:{len(cals)}", "ok")
        else:
            self._set_chip(self.chip_cal, "Cal:0", "warn")

        if len(strats) > 0:
            self._set_chip(self.chip_st, f"Str:{len(strats)}", "ok")
        else:
            self._set_chip(self.chip_st, "Str:0", "warn")

        # Map overlay chip (safe if key missing)
        ov = "both"
        if self._overlay_key:
            ov = _as_str(s.get(self._overlay_key, "both")) or "both"
        ov = ov.lower()

        if ov in {"a", "city_a"}:
            self._set_chip(self.chip_ov, "OV:A", "ok")
        elif ov in {"b", "city_b"}:
            self._set_chip(self.chip_ov, "OV:B", "ok")
        else:
            self._set_chip(self.chip_ov, "OV:A+B", "ok")

        self._fit_chips()

    # -------------------------------------------------
    # UI
    # -------------------------------------------------
    def _build_ui(self) -> None:
        self.setObjectName("trainTopBar")

        root = QHBoxLayout(self)
        root.setContentsMargins(8, 4, 8, 4)
        root.setSpacing(8)

        self.lbl_title = QLabel("Transfer matrix", self)
        self.lbl_title.setObjectName("inferHeadTitle")
        self.lbl_title.setSizePolicy(
            QSizePolicy.Fixed,
            QSizePolicy.Fixed,
        )

        self.lbl_plan = _ElideLabel("", self)
        self.lbl_plan.setObjectName("sumLine")
        self.lbl_plan.setWordWrap(False)
        self.lbl_plan.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Fixed,
        )

        # Chips (auto-hidden by _fit_chips)
        self.chip_has = _Chip("—", kind="off", parent=self)
        self.chip_ab = _Chip("A/B", kind="off", parent=self)
        self.chip_sp = _Chip("S:0", kind="warn", parent=self)
        self.chip_cal = _Chip("Cal:0", kind="warn", parent=self)
        self.chip_st = _Chip("Str:0", kind="warn", parent=self)
        self.chip_ov = _Chip("OV:A+B", kind="ok", parent=self)
        self.chip_ov.setVisible(False)

        self._chips = [
            self.chip_has,
            self.chip_ab,
            self.chip_sp,
            self.chip_cal,
            self.chip_st,
            self.chip_ov,
        ]

        # Segmented Run/Map (prominent)
        self.seg = QWidget(self)
        self.seg.setObjectName("xferSeg")

        seg_l = QHBoxLayout(self.seg)
        seg_l.setContentsMargins(2, 2, 2, 2)
        seg_l.setSpacing(0)

        self.btn_run_mode = QToolButton(self.seg)
        self.btn_run_mode.setText("Run")
        self.btn_run_mode.setCheckable(True)
        self.btn_run_mode.setObjectName("xferSegBtn")
        self.btn_run_mode.setProperty("pos", "left")

        self.btn_map_mode = QToolButton(self.seg)
        self.btn_map_mode.setText("Map")
        self.btn_map_mode.setCheckable(True)
        self.btn_map_mode.setObjectName("xferSegBtn")
        self.btn_map_mode.setProperty("pos", "right")

        seg_l.addWidget(self.btn_run_mode)
        seg_l.addWidget(self.btn_map_mode)

        # Filter + Search + Help (same affordance as Inference)
        self.btn_filter = self._mk_icon_btn(
            icon_names=("filter2.svg", "filter.svg"),
            fallback_text="🔎",
            tip="Filter / search",
            fallback_std=QStyle.SP_FileDialogContentsView,
        )
        self.btn_filter.setObjectName("miniAction")

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
        self.btn_help.setObjectName("miniAction")

        # Layout
        root.addWidget(self.lbl_title, 0)
        root.addWidget(self.lbl_plan, 1)

        for c in self._chips:
            root.addWidget(c, 0)

        root.addSpacing(6)
        root.addWidget(self.seg, 0)
        root.addSpacing(4)
        root.addWidget(self.btn_filter, 0)
        root.addWidget(self.ed_search, 0)
        root.addWidget(self.btn_help, 0)

        # default mode
        self.set_mode("run")

    def _set_chip(self, chip: QLabel, txt: str, kind: str) -> None:
        chip.setText(str(txt))
        chip.setProperty("kind", str(kind))
        chip.style().unpolish(chip)
        chip.style().polish(chip)

    def _fit_chips(self) -> None:
        for c in self._chips:
            c.show()

        # always keep these if possible
        must_keep = {self.chip_has, self.chip_ab}

        # rough budget: leave room for right controls
        budget = self.width() - 340
        used = self.lbl_title.sizeHint().width()
        used += self.seg.sizeHint().width()

        for c in self._chips:
            if not c.isVisible():
                continue
            used += c.sizeHint().width() + 6

        if used <= budget:
            return

        # hide from the end first (least important)
        for c in reversed(self._chips):
            if c in must_keep:
                continue
            if not c.isVisible():
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

    # -------------------------------------------------
    # Wiring
    # -------------------------------------------------
    def _wire(self) -> None:
        self.btn_run_mode.clicked.connect(
            lambda: self.mode_changed.emit("run")
        )
        self.btn_map_mode.clicked.connect(
            lambda: self.mode_changed.emit("map")
        )

        self.btn_help.clicked.connect(self.help_clicked.emit)
        self.btn_filter.clicked.connect(self.filter_clicked.emit)
        self.ed_search.textChanged.connect(self.search_changed.emit)

        self._s.config_changed.connect(lambda _k: self.refresh_from_store())
