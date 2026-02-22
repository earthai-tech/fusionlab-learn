# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>

"""geoprior.ui.setup.cards.time

Time window & forecast card.

Modern UX:
- Timeline preview (train end -> start -> end).
- Validation hints and badges.
- Quick presets for horizon and steps.
- Calendar preview for train/forecast ranges.
- Store-driven via Binder.
"""

from __future__ import annotations

from typing import Optional

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QCheckBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from .base import CardBase
from ..bindings import Binder
from ....config.store import GeoConfigStore


class TimeWindowCard(CardBase):
    """Time window & forecast (store-driven)."""

    def __init__(
        self,
        *,
        store: GeoConfigStore,
        binder: Binder,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(
            section_id="time",
            title="Time window & forecast",
            subtitle=(
                "Define training cutoff, forecast start, "
                "and horizon. A preview updates live."
            ),
            parent=parent,
        )

        self.store = store
        self.binder = binder

        self._build()
        self._wire()
        self.refresh()

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

        self.grp_core = self._build_core(grid)
        self.grp_tools = self._build_tools(grid)

        g.addWidget(self.grp_core, 0, 0)
        g.addWidget(self.grp_tools, 0, 1)

        body.addWidget(grid, 0)

    def _build_core(self, parent: QWidget) -> QGroupBox:
        box = QGroupBox("Window", parent)
        lay = QVBoxLayout(box)
        lay.setContentsMargins(10, 10, 10, 10)
        lay.setSpacing(10)

        self.timeline = QWidget(box)
        tl = QHBoxLayout(self.timeline)
        tl.setContentsMargins(0, 0, 0, 0)
        tl.setSpacing(8)

        self.lbl_train = self._pill("Train end", box)
        self.lbl_arrow1 = QLabel("→", box)
        self.lbl_start = self._pill("Forecast start", box)
        self.lbl_arrow2 = QLabel("→", box)
        self.lbl_end = self._pill("Forecast end", box)

        tl.addWidget(self.lbl_train, 0)
        tl.addWidget(self.lbl_arrow1, 0)
        tl.addWidget(self.lbl_start, 0)
        tl.addWidget(self.lbl_arrow2, 0)
        tl.addWidget(self.lbl_end, 0)
        tl.addStretch(1)

        lay.addWidget(self.timeline, 0)

        form = QWidget(box)
        f = QGridLayout(form)
        f.setContentsMargins(0, 0, 0, 0)
        f.setHorizontalSpacing(10)
        f.setVerticalSpacing(8)
        f.setColumnStretch(1, 1)
        f.setColumnStretch(3, 1)

        self.sp_train_end = QSpinBox(box)
        self.sp_train_end.setRange(1900, 2200)

        self.sp_fc_start = QSpinBox(box)
        self.sp_fc_start.setRange(1900, 2200)

        self.sp_horizon = QSpinBox(box)
        self.sp_horizon.setRange(1, 100)

        self.sp_steps = QSpinBox(box)
        self.sp_steps.setRange(1, 500)

        f.addWidget(QLabel("Train end year:", box), 0, 0)
        f.addWidget(self.sp_train_end, 0, 1)

        f.addWidget(QLabel("Forecast start:", box), 0, 2)
        f.addWidget(self.sp_fc_start, 0, 3)

        f.addWidget(QLabel("Horizon (years):", box), 1, 0)
        f.addWidget(self.sp_horizon, 1, 1)

        f.addWidget(QLabel("Time steps:", box), 1, 2)
        f.addWidget(self.sp_steps, 1, 3)

        lay.addWidget(form, 0)

        self.lbl_hint = QLabel("", box)
        self.lbl_hint.setWordWrap(True)
        self.lbl_hint.setObjectName("timeHint")

        lay.addWidget(self.lbl_hint, 0)

        box.setStyleSheet(
            "\n".join(
                [
                    "QLabel#timeHint {",
                    "  color: rgba(30,30,30,0.72);",
                    "  font-size: 11px;",
                    "}",
                ]
            )
        )

        # Bindings
        self.binder.bind_spin_box(
            "train_end_year",
            self.sp_train_end,
        )
        self.binder.bind_spin_box(
            "forecast_start_year",
            self.sp_fc_start,
        )
        self.binder.bind_spin_box(
            "forecast_horizon_years",
            self.sp_horizon,
        )
        self.binder.bind_spin_box(
            "time_steps",
            self.sp_steps,
        )

        return box

    def _build_tools(self, parent: QWidget) -> QGroupBox:
        box = QGroupBox("Options", parent)
        lay = QVBoxLayout(box)
        lay.setContentsMargins(10, 10, 10, 10)
        lay.setSpacing(10)

        self.chk_future_npz = QCheckBox(
            "Build future NPZ",
            box,
        )
        self.binder.bind_checkbox(
            "build_future_npz",
            self.chk_future_npz,
        )
        lay.addWidget(self.chk_future_npz, 0)

        lay.addWidget(self._build_presets(box), 0)
        lay.addWidget(self._build_calendar(box), 0)

        self.badge(
            "status",
            text="OK",
            accent="ok",
            tip="Time window consistency",
        )

        return box

    def _build_presets(self, parent: QWidget) -> QGroupBox:
        grp = QGroupBox("Quick presets", parent)
        g = QGridLayout(grp)
        g.setContentsMargins(10, 10, 10, 10)
        g.setHorizontalSpacing(8)
        g.setVerticalSpacing(8)

        g.addWidget(QLabel("Horizon:", grp), 0, 0)
        g.addWidget(self._preset_btn("1y", 1), 0, 1)
        g.addWidget(self._preset_btn("3y", 3), 0, 2)
        g.addWidget(self._preset_btn("5y", 5), 0, 3)
        g.addWidget(self._preset_btn("10y", 10), 0, 4)

        g.addWidget(QLabel("Steps:", grp), 1, 0)
        g.addWidget(self._steps_btn("3", 3), 1, 1)
        g.addWidget(self._steps_btn("5", 5), 1, 2)
        g.addWidget(self._steps_btn("8", 8), 1, 3)
        g.addWidget(self._steps_btn("12", 12), 1, 4)

        return grp

    def _build_calendar(self, parent: QWidget) -> QGroupBox:
        grp = QGroupBox("Calendar preview", parent)
        lay = QVBoxLayout(grp)
        lay.setContentsMargins(10, 10, 10, 10)
        lay.setSpacing(8)

        self.row_train = self._kv_row(grp, "Train")
        self.row_fc = self._kv_row(grp, "Forecast")
        self.row_meta = self._kv_row(grp, "Summary")

        lay.addWidget(self.row_train["w"], 0)
        lay.addWidget(self.row_fc["w"], 0)
        lay.addWidget(self.row_meta["w"], 0)

        self.cal_train = self.row_train["v"]
        self.cal_fc = self.row_fc["v"]
        self.cal_meta = self.row_meta["v"]

        self._calendar_style(grp)

        return grp

    def _calendar_style(self, grp: QGroupBox) -> None:
        grp.setStyleSheet(
            "\n".join(
                [
                    "QLabel#calKey {",
                    "  color: rgba(30,30,30,0.72);",
                    "  font-size: 11px;",
                    "}",
                    "QLabel#calVal {",
                    "  font-weight: 600;",
                    "  padding: 3px 10px;",
                    "  border-radius: 12px;",
                    "  border: 1px solid",
                    "    rgba(0,0,0,0.12);",
                    "  background: rgba(0,0,0,0.03);",
                    "}",
                ]
            )
        )

    def _kv_row(
        self,
        parent: QWidget,
        key: str,
    ) -> dict:
        w = QWidget(parent)
        lay = QHBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(8)

        k = QLabel(f"{key}:", w)
        k.setObjectName("calKey")

        v = QLabel("-", w)
        v.setObjectName("calVal")
        v.setTextInteractionFlags(
            Qt.TextSelectableByMouse
        )

        lay.addWidget(k, 0)
        lay.addWidget(v, 1)
        return {"w": w, "v": v}

    def _pill(self, label: str, parent: QWidget) -> QLabel:
        lab = QLabel(str(label), parent)
        lab.setObjectName("timePill")
        lab.setTextInteractionFlags(
            Qt.TextSelectableByMouse
        )
        lab.setAlignment(Qt.AlignCenter)
        lab.setStyleSheet(
            "\n".join(
                [
                    "QLabel#timePill {",
                    "  padding: 3px 10px;",
                    "  border-radius: 12px;",
                    "  border: 1px solid",
                    "    rgba(0,0,0,0.12);",
                    "  background: rgba(0,0,0,0.03);",
                    "  font-weight: 600;",
                    "}",
                ]
            )
        )
        return lab

    def _preset_btn(self, text: str, years: int) -> QPushButton:
        b = QPushButton(str(text), self)
        b.setObjectName("miniAction")
        b.setCursor(Qt.PointingHandCursor)
        b.clicked.connect(lambda: self._set_horizon(years))
        return b

    def _steps_btn(self, text: str, steps: int) -> QPushButton:
        b = QPushButton(str(text), self)
        b.setObjectName("miniAction")
        b.setCursor(Qt.PointingHandCursor)
        b.clicked.connect(lambda: self._set_steps(steps))
        return b

    # -----------------------------------------------------------------
    # Wiring
    # -----------------------------------------------------------------
    def _wire(self) -> None:
        self.sp_train_end.valueChanged.connect(self._update)
        self.sp_fc_start.valueChanged.connect(self._update)
        self.sp_horizon.valueChanged.connect(self._update)
        self.sp_steps.valueChanged.connect(self._update)

        self.store.config_changed.connect(
            lambda _k: self.refresh(),
        )
        self.store.config_replaced.connect(
            lambda _cfg: self.refresh(),
        )

    # -----------------------------------------------------------------
    # Actions
    # -----------------------------------------------------------------
    def _set_horizon(self, years: int) -> None:
        self.store.patch({"forecast_horizon_years": int(years)})

    def _set_steps(self, steps: int) -> None:
        self.store.patch({"time_steps": int(steps)})

    def _update(self) -> None:
        self._sync_preview()

    # -----------------------------------------------------------------
    # Refresh
    # -----------------------------------------------------------------
    def refresh(self) -> None:
        self._sync_preview()

    def _sync_preview(self) -> None:
        te = int(self.sp_train_end.value())
        fs = int(self.sp_fc_start.value())
        hz = int(self.sp_horizon.value())
        st = int(self.sp_steps.value())

        fe = fs + hz - 1

        self.lbl_train.setText(f"Train end: {te}")
        self.lbl_start.setText(f"Forecast start: {fs}")
        self.lbl_end.setText(f"Forecast end: {fe}")

        self.cal_train.setText(f"…–{te}")
        self.cal_fc.setText(f"{fs}–{fe}")
        self.cal_meta.setText(f"{hz}y, {st} steps")

        ok = bool(fs > te)
        if ok:
            msg = (
                "Looks good. Forecast starts after "
                "training period."
            )
            self.badge(
                "status",
                text="OK",
                accent="ok",
                tip="Valid window",
            )
        else:
            msg = (
                "Forecast start should be greater than "
                "train end year."
            )
            self.badge(
                "status",
                text="Check",
                accent="warn",
                tip="Invalid window",
            )

        self.lbl_hint.setText(msg)

        self.badge(
            "end",
            text=f"End {fe}",
            accent="ok" if ok else "warn",
            tip="Computed forecast end year",
        )
