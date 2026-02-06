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

from PyQt5.QtCore import Qt, pyqtSignal, QEvent, QPoint
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QStyle,
    QToolButton,
    QVBoxLayout,
    QWidget,
    QScrollArea, 
    QSplitter,
    QGraphicsDropShadowEffect
)

from ..icon_utils import try_icon
from .keys import (
    MAP_SELECT_IDS,
    MAP_SELECT_MODE,
    MAP_SELECT_OPEN,
    MAP_SELECT_PINNED,
    MAP_SELECT_POS_X,
    MAP_SELECT_POS_Y,
    MAP_SELECT_MANUAL,
)

from .selection_plot import SelectionPlot

class SelectionPanel(QFrame):
    """Right-side insights drawer for current selection."""

    request_close = pyqtSignal()
    request_pin = pyqtSignal(bool)
    request_relayout = pyqtSignal()

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

        self.setAttribute(Qt.WA_TranslucentBackground, True)
        
        wrap = QVBoxLayout(self)
        wrap.setContentsMargins(14, 14, 14, 14)
        wrap.setSpacing(0)
        
        self.card = QFrame(self)
        self.card.setObjectName("gpSelCard")
        self.card.setFrameShape(QFrame.NoFrame)
        wrap.addWidget(self.card, 1)
        
        root = QVBoxLayout(self.card)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)
        
        eff = QGraphicsDropShadowEffect(self.card)
        eff.setBlurRadius(26.0)
        eff.setOffset(0.0, 6.0)
        eff.setColor(QColor(0, 0, 0, 85))
        self.card.setGraphicsEffect(eff)

        self.drag_bar = QWidget(self.card)
        self.drag_bar.setObjectName("gpSelDragBar")
        self.drag_bar.setCursor(Qt.OpenHandCursor)
        self.drag_bar.setAttribute(Qt.WA_Hover, True)
        self.drag_bar.setMouseTracking(True)

        head = QHBoxLayout(self.drag_bar)
        head.setContentsMargins(0, 0, 0, 0)
        head.setSpacing(8)
        
        self.lb_title = QLabel("Selection", self.drag_bar)
        self.lb_title.setObjectName("gpSelTitle")
        self.lb_title.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        
        self.btn_pin = QToolButton(self.drag_bar)
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
        
        self.btn_close = QToolButton(self.drag_bar)
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
        
        root.addWidget(self.drag_bar, 0)

        # root.addLayout(head, 0)
        
        self.body = QSplitter(Qt.Horizontal, self)
        self.body.setChildrenCollapsible(False)

        self._details = QWidget(self.body)
        dl = QVBoxLayout(self._details)
        dl.setContentsMargins(0, 0, 0, 0)
        dl.setSpacing(8)

        self.lb_hint = QLabel(
            "Select a point or a group\n"
            "to see insights here.",
            self,
        )
        self.lb_hint.setObjectName("gpSelHint")
        self.lb_hint.setAlignment(Qt.AlignCenter)

        self.lb_hint.setMinimumHeight(120)
        self.lb_hint.setWordWrap(True)
        dl.addWidget(self.lb_hint, 0)

        self.lb_summary = QLabel("", self._details)
        self.lb_summary.setObjectName("gpSelSummary")
        self.lb_summary.setWordWrap(True)
        self.lb_summary.setTextFormat(Qt.RichText)
        self.lb_summary.setAlignment(
            Qt.AlignLeft | Qt.AlignTop
        )
        self.lb_summary.setVisible(False)
        dl.addWidget(self.lb_summary, 0)

        self.lb_busy = QLabel("", self)
        self.lb_busy.setObjectName("gpSelBusy")
        self.lb_busy.setAlignment(
            Qt.AlignCenter
        )
        self.lb_busy.setVisible(False)
        dl.addWidget(self.lb_busy, 0)

        self.details_scroll = QScrollArea(self.body)
        self.details_scroll.setObjectName("gpSelDetails")
        self.details_scroll.viewport().setObjectName(
            "gpSelDetailsVp"
        )

        self.details_scroll.setFrameShape(QFrame.NoFrame)
        self.details_scroll.setWidgetResizable(True)
        self.details_scroll.setWidget(self._details)

        self.plot = SelectionPlot(parent=self.body)
        self.plot.setVisible(False)

        self.body.addWidget(self.details_scroll)
        self.body.addWidget(self.plot)
        self.body.setStretchFactor(0, 1)
        self.body.setStretchFactor(1, 2)
        
        self.details_scroll.setMinimumWidth(240)
        self.plot.setMinimumWidth(280)


        root.addWidget(self.body, 1)
        
        self._dragging = False
        self._drag_off = QPoint(0, 0)
        
        self.drag_bar.installEventFilter(self)

        self._connect()
        self._sync_from_store()
        
    def eventFilter(self, obj, ev) -> bool:
        if obj is getattr(self, "drag_bar", None):
            et = ev.type()
    
            if et == QEvent.MouseButtonPress:
                if ev.button() == Qt.LeftButton:
                    # Don't drag when clicking buttons
                    child = self.drag_bar.childAt(ev.pos())
                    if child in (self.btn_pin, self.btn_close):
                        return False
    
                    self._dragging = True
                    self.drag_bar.setCursor(Qt.ClosedHandCursor)
                    self.raise_()
    
                    # global -> widget offset
                    try:
                        self._drag_off = ev.globalPos() - self.pos()
                    except Exception:
                        self._drag_off = QPoint(0, 0)
                    return True
    
            if et == QEvent.MouseMove:
                if self._dragging and (ev.buttons() & Qt.LeftButton):
                    try:
                        p = ev.globalPos() - self._drag_off
                    except Exception:
                        return True
    
                    par = self.parentWidget()
                    if par is not None:
                        r = par.rect()
                        x = max(10, min(p.x(), r.width() - self.width() - 10))
                        y = max(56, min(p.y(), r.height() - self.height() - 10))
                        self.move(int(x), int(y))
                    else:
                        self.move(int(p.x()), int(p.y()))
                    return True
    
            if et == QEvent.MouseButtonRelease:
                if self._dragging:
                    self._dragging = False
                    self.drag_bar.setCursor(Qt.OpenHandCursor)
    
                    # Persist manual position
                    if self._s is not None:
                        with self._s.batch():
                            self._s.set(MAP_SELECT_MANUAL, True)
                            self._s.set(MAP_SELECT_POS_X, int(self.x()))
                            self._s.set(MAP_SELECT_POS_Y, int(self.y()))
    
                    self.request_relayout.emit()
                    return True
    
        return super().eventFilter(obj, ev)

    def _autofit_body(self) -> None:
        b = getattr(self, "body", None)
        if b is None:
            return
    
        total = int(b.width())
        if total <= 0:
            return
    
        plot_min = int(getattr(self.plot, "minimumWidth", lambda: 280)())
        plot_min = max(260, plot_min)
    
        hint = 240
        if self.lb_summary.isVisible():
            hint = max(hint, int(
                self.lb_summary.sizeHint().width()))
        if self.lb_hint.isVisible():
            hint = max(hint, int(
                self.lb_hint.sizeHint().width()))
    
        # margins + some breathing room
        want = hint + 24
    
        left_max = max(240, total - plot_min)
        left = max(240, min(want, left_max))
        right = max(plot_min, total - left)
    
        b.setSizes([left, right])

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
        # self.lb_summary.setText(f"Error:\n{m}")
        self.lb_summary.setText(
            f"<b>Error</b><br>{m}"
        )
        
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

        rows = []
        if mode == "point" and ids:
            rows.append(("Point", ids[0]))
        elif mode == "group":
            rows.append(("Group", f"{len(ids)} points"))

        if n is not None:
            rows.append(("n_points", n))
        if t0 is not None or t1 is not None:
            rows.append(("t", f"{t0} → {t1}"))
        if z0 is not None or z1 is not None:
            rows.append(("z", f"{z0} → {z1}"))
        if zm is not None:
            rows.append(("z_mean", zm))

        cur = res.get("current", None)
        if isinstance(cur, dict):
            # lines.append("")
            # lines.append("Current frame:")
            # for k in ("n", "min", "max", "mean", "median"):
            #     if k in cur:
            #         lines.append(f"  {k}: {cur[k]}")
            rows.append(("", ""))
            rows.append(("Current frame", ""))
            for k in ("n", "min", "max", "mean", "median"):
                if k in cur:
                    rows.append((k, cur[k]))

        rp = res.get("risk_p", None)
        if rp is not None:
        #     lines.append("")
        #     lines.append(f"Risk P(Z > thr): {rp}")

        # self.lb_summary.setText("\n".join(lines))
            rows.append(("", ""))
            rows.append(("Risk P(Z > thr)", rp))

        html = ["<table cellspacing='0' cellpadding='2'>"]
        for k, v in rows:
            if k == "" and v == "":
                html.append(
                    "<tr><td colspan='2'><br></td></tr>"
                )
                continue
            if k and (v == ""):
                html.append(
                    "<tr><td colspan='2'>"
                    f"<b>{k}</b>"
                    "</td></tr>"
                )
                continue
            html.append(
                "<tr>"
                f"<td><b>{k}</b></td>"
                f"<td>{v}</td>"
                "</tr>"
            )
        html.append("</table>")
        
        self.lb_summary.setText("".join(html))

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
                
                overlay = res.get("overlay", None)
                
                band = res.get("band", None)
                if (
                    isinstance(band, (list, tuple))
                    and len(band) == 2
                ):
                    band = (str(band[0]), str(band[1]))
                else:
                    band = None
                    
                self.plot.plot_point(
                    d1,
                    t_col=t_col,
                    z_col=z_col,
                    band=band,
                    overlay=overlay,
                )
            else:
                self.plot.setVisible(False)
                self.plot.clear("No series")
        
        elif mode == "group":
            tr = res.get("trend", None)
            if isinstance(tr, pd.DataFrame) and (not tr.empty):
                self.plot.setVisible(True)
                # self.plot.plot_group(tr, t_col=t_col)
                self.plot.plot_group(
                    tr,
                    t_col=t_col,
                    y_label=z_col,
                )
            else:
                self.plot.setVisible(False)
                self.plot.clear("No trend")
        
        else:
            self.plot.setVisible(False)
            self.plot.clear("")

        self._autofit_body()
        self.adjustSize()

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
        except:
            pass
        self.lb_summary.setTextInteractionFlags(
            Qt.TextSelectableByMouse
        )

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
