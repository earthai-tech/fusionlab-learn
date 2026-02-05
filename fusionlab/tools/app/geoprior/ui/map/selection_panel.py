# geoprior/ui/map/selection_panel.py
# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio

"""geoprior.ui.map.selection_panel

Selection insights drawer (right-side).

This is a UI skeleton only.
We will later plug in:
  - point details + plots
  - group summaries + distributions + trends
  - background stats worker
"""

from __future__ import annotations

from typing import Optional, Sequence
import pandas as pd 

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QStyle,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ..icon_utils import try_icon
from .keys import (
    MAP_SELECT_IDS,
    MAP_SELECT_MODE,
    MAP_SELECT_OPEN,
    MAP_SELECT_PINNED,
)

from .selection_plot import SelectionPlot

class SelectionPanel(QFrame):
    """Right-side insights drawer for current selection."""

    request_close = pyqtSignal()
    request_pin = pyqtSignal(bool)

    def __init__(
        self,
        *,
        store=None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._s = store

        self.setObjectName("gpSelectionPanel")
        self.setFrameShape(QFrame.NoFrame)

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        head = QHBoxLayout()
        head.setContentsMargins(0, 0, 0, 0)
        head.setSpacing(8)

        self.lb_title = QLabel("Selection", self)
        self.lb_title.setObjectName("gpSelTitle")
        self.lb_title.setAlignment(
            Qt.AlignLeft | Qt.AlignVCenter
        )

        self.btn_pin = QToolButton(self)
        self.btn_pin.setObjectName("miniAction")
        self.btn_pin.setProperty("role", "mapHead")
        self.btn_pin.setAutoRaise(True)
        self.btn_pin.setCheckable(True)
        self.btn_pin.setToolTip("Pin panel")
        self._set_icon(
            self.btn_pin,
            "pin.svg",
            QStyle.SP_DialogYesButton,
        )

        self.btn_close = QToolButton(self)
        self.btn_close.setObjectName("miniAction")
        self.btn_close.setProperty("role", "mapHead")
        self.btn_close.setAutoRaise(True)
        self.btn_close.setToolTip("Close")
        self._set_icon(
            self.btn_close,
            "close2.svg",
            QStyle.SP_DockWidgetCloseButton,
        )

        head.addWidget(self.lb_title, 1)
        head.addWidget(self.btn_pin, 0)
        head.addWidget(self.btn_close, 0)

        root.addLayout(head, 0)

        self.lb_hint = QLabel(
            "Select a point or a group\n"
            "to see insights here.",
            self,
        )
        self.lb_hint.setObjectName("gpSelHint")
        self.lb_hint.setAlignment(Qt.AlignCenter)
        self.lb_hint.setMinimumHeight(140)
        root.addWidget(self.lb_hint, 1)
        
        self.plot = SelectionPlot(parent=self)
        self.plot.setVisible(False)
        root.addWidget(self.plot, 1)

        self.lb_summary = QLabel("", self)
        self.lb_summary.setObjectName("gpSelSummary")
        self.lb_summary.setWordWrap(True)
        self.lb_summary.setAlignment(
            Qt.AlignLeft | Qt.AlignTop
        )
        self.lb_summary.setVisible(False)
        root.addWidget(self.lb_summary, 0)

        self.lb_busy = QLabel("", self)
        self.lb_busy.setObjectName("gpSelBusy")
        self.lb_busy.setAlignment(
            Qt.AlignCenter
        )
        self.lb_busy.setVisible(False)
        root.addWidget(self.lb_busy, 0)

        self._connect()
        self._sync_from_store()
        
    def set_busy(self, on: bool) -> None:
        on = bool(on)
        self.lb_busy.setVisible(on)
        if on:
            self.lb_busy.setText("Computing…")
        else:
            self.lb_busy.setText("")

    def set_error(self, msg: str) -> None:
        m = str(msg or "").strip()
        if not m:
            return
        self.lb_summary.setVisible(True)
        self.lb_hint.setVisible(False)
        self.lb_summary.setText(f"Error:\n{m}")
        
        if getattr(self, "plot", None) is not None:
            self.plot.setVisible(False)
            self.plot.clear("Error")


    def set_result(self, res: object) -> None:
        if not isinstance(res, dict):
            self.lb_summary.setVisible(False)
            self.lb_hint.setVisible(True)
            return

        mode = str(res.get("mode", "off")).strip().lower()
        ids = list(res.get("ids", []) or [])
        summ = res.get("summary", {}) or {}

        if res.get("empty", False):
            self.lb_summary.setVisible(False)
            self.lb_hint.setVisible(True)
            self.set_selection(mode, ids)
            return

        n = summ.get("n_points", None)
        t0 = summ.get("t_min", None)
        t1 = summ.get("t_max", None)
        z0 = summ.get("z_min", None)
        z1 = summ.get("z_max", None)
        zm = summ.get("z_mean", None)

        lines = []

        if mode == "point" and ids:
            lines.append(f"Point: {ids[0]}")
        elif mode == "group":
            lines.append(f"Group: {len(ids)} points")

        if n is not None:
            lines.append(f"n_points: {n}")
        if t0 is not None or t1 is not None:
            lines.append(f"t: {t0} → {t1}")
        if z0 is not None or z1 is not None:
            lines.append(f"z: {z0} → {z1}")
        if zm is not None:
            lines.append(f"z_mean: {zm}")

        cur = res.get("current", None)
        if isinstance(cur, dict):
            lines.append("")
            lines.append("Current frame:")
            for k in ("n", "min", "max", "mean", "median"):
                if k in cur:
                    lines.append(f"  {k}: {cur[k]}")

        rp = res.get("risk_p", None)
        if rp is not None:
            lines.append("")
            lines.append(f"Risk P(Z > thr): {rp}")

        self.lb_summary.setText("\n".join(lines))
        self.lb_summary.setVisible(True)
        self.lb_hint.setVisible(False)
        
        t_col = str(res.get("t_col", "t") or "t")
        z_col = str(res.get("z_col", "v") or "v")
        
        if getattr(self, "plot", None) is None:
            return
        
        if mode == "point":
            d1 = res.get("series", None)
            if isinstance(d1, pd.DataFrame) and (not d1.empty):
                self.plot.setVisible(True)
                self.plot.plot_point(
                    d1,
                    t_col=t_col,
                    z_col=z_col,
                    band=None,
                )
            else:
                self.plot.setVisible(False)
                self.plot.clear("No series")
        
        elif mode == "group":
            tr = res.get("trend", None)
            if isinstance(tr, pd.DataFrame) and (not tr.empty):
                self.plot.setVisible(True)
                self.plot.plot_group(tr, t_col=t_col)
            else:
                self.plot.setVisible(False)
                self.plot.clear("No trend")
        
        else:
            self.plot.setVisible(False)
            self.plot.clear("")


    def set_selection(
        self,
        mode: str,
        ids: Sequence[int],
    ) -> None:
        """Update the placeholder state (no plots yet)."""
        m = str(mode or "off").strip().lower()
        n = len(list(ids or []))
        if m == "point" and n == 1:
            self.lb_hint.setText(
                f"Point selected: {list(ids)[0]}"
            )
        elif m == "group" and n:
            self.lb_hint.setText(
                f"Group selected: {n} points"
            )
        else:
            self.lb_hint.setText(
                "Select a point or a group\n"
                "to see insights here."
            )

    def _connect(self) -> None:
        self.btn_close.clicked.connect(
            lambda: self.request_close.emit()
        )
        self.btn_pin.toggled.connect(self._on_pin)

        if self._s is None:
            return
        try:
            self._s.config_changed.connect(
                self._on_store_changed
            )
        except Exception:
            pass

    def _on_pin(self, on: bool) -> None:
        if self._s is not None:
            self._s.set(MAP_SELECT_PINNED, bool(on))
        self.request_pin.emit(bool(on))

    def _on_store_changed(self, keys) -> None:
        ks = set(keys or [])
        if not ks:
            return
        watch = {
            MAP_SELECT_MODE,
            MAP_SELECT_IDS,
            MAP_SELECT_OPEN,
            MAP_SELECT_PINNED,
        }
        if not ks.intersection(watch):
            return
        self._sync_from_store()

    def _sync_from_store(self) -> None:
        if self._s is None:
            return

        m = self._s.get(MAP_SELECT_MODE, "off")
        ids = self._s.get(MAP_SELECT_IDS, []) or []
        open_ = bool(self._s.get(MAP_SELECT_OPEN, False))
        pin = bool(self._s.get(MAP_SELECT_PINNED, False))

        was = self.btn_pin.blockSignals(True)
        self.btn_pin.setChecked(pin)
        self.btn_pin.blockSignals(was)

        self.setVisible(open_ or pin)
        self.set_selection(str(m), ids)

    @staticmethod
    def _set_icon(
        btn: QToolButton,
        name: str,
        sp: int,
    ) -> None:
        fb = btn.style().standardIcon(int(sp))
        btn.setIcon(try_icon(name, fallback=fb, size=18))
