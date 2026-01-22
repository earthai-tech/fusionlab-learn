# ui/train/run_preview.py
# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>

"""
Train Run Preview panel.

Contains:
- Run plan text (selectable)
- Run plan visual (timeline + lambda bars)
- Compute summary (OS/Python/CPU/RAM/GPU/backend)
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from PyQt5.QtCore import Qt, QRectF, pyqtSignal
from PyQt5.QtGui import QColor, QPainter, QPen, QPainterPath
from PyQt5.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QStyle,
    QToolButton,
    QVBoxLayout,
    QWidget,
    QSizePolicy,
)

from ..icon_utils import try_icon
from ...device_options import runtime_summary_text
from ...config.store import GeoConfigStore


__all__ = ["RunPreviewPanel", "RunPreviewViz"]


class RunPreviewPanel(QFrame):
    """
    Right-side preview widget (content only).

    This is meant to be placed inside a "Run preview" card body.
    It does NOT create a card header itself.

    Public API
    ----------
    - set_plan_text(text)
    - set_plan(**kwargs)  -> forwards to RunPreviewViz
    - refresh_compute()
    - refresh_all()
    """

    toast = pyqtSignal(str)

    def __init__(
        self,
        *,
        store: GeoConfigStore,
        show_compute: bool = True,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._store = store
        self._plan_text = ""
        self._show_compute = bool(show_compute)

        self.setObjectName("runPreviewPanel")
        self.setFrameShape(QFrame.NoFrame)
        self.setAttribute(Qt.WA_StyledBackground, True)

        self._build_ui()
        self._wire_store()
        self.refresh_all()

    # -------------------------------------------------
    # Icons
    # -------------------------------------------------
    def _std_icon(
        self,
        sp: QStyle.StandardPixmap,
    ):
        return self.style().standardIcon(sp)

    def _set_icon(
        self,
        btn: QToolButton,
        name: str,
        fallback: QStyle.StandardPixmap,
    ) -> None:
        ic = try_icon(name)
        if ic is None:
            ic = self._std_icon(fallback)
        btn.setIcon(ic)

    def _mk_icon_btn(
        self,
        tip: str,
        icon_name: str,
        fallback: QStyle.StandardPixmap,
    ) -> QToolButton:
        b = QToolButton(self)
        b.setAutoRaise(True)
        b.setToolButtonStyle(Qt.ToolButtonIconOnly)
        b.setToolTip(tip)
        self._set_icon(b, icon_name, fallback)
        return b

    # -------------------------------------------------
    # UI
    # -------------------------------------------------
    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(10)
    
        # Run plan (title + copy)
        plan_hdr = QHBoxLayout()
        plan_hdr.setContentsMargins(0, 0, 0, 0)
        plan_hdr.setSpacing(8)
    
        plan_t = QLabel("Run plan")
        plan_t.setObjectName("subTitle")
    
        self.btn_copy = self._mk_icon_btn(
            "Copy run plan",
            "copy.svg",
            QStyle.SP_DialogSaveButton,
        )
    
        plan_hdr.addWidget(plan_t, 0)
        plan_hdr.addStretch(1)
        plan_hdr.addWidget(self.btn_copy, 0)
        root.addLayout(plan_hdr)
    
        self.lbl_plan = QLabel("")
        self.lbl_plan.setObjectName("runPlanText")
        self.lbl_plan.setWordWrap(True)
        self.lbl_plan.setTextInteractionFlags(
            Qt.TextSelectableByMouse
        )
        root.addWidget(self.lbl_plan, 0)
    
        # Run plan visual
        self.viz = RunPreviewViz(self)
        root.addWidget(self.viz, 1)
    
        # Optional Compute block
        if self._show_compute:
            comp_hdr = QHBoxLayout()
            comp_hdr.setContentsMargins(0, 0, 0, 0)
            comp_hdr.setSpacing(8)
    
            comp_t = QLabel("Compute")
            comp_t.setObjectName("subTitle")
    
            self.btn_refresh = self._mk_icon_btn(
                "Refresh compute info",
                "refresh.svg",
                QStyle.SP_BrowserReload,
            )
    
            comp_hdr.addWidget(comp_t, 0)
            comp_hdr.addStretch(1)
            comp_hdr.addWidget(self.btn_refresh, 0)
            root.addLayout(comp_hdr)
    
            self.lbl_compute = QLabel("")
            self.lbl_compute.setObjectName("runComputeText")
            self.lbl_compute.setWordWrap(True)
            self.lbl_compute.setTextInteractionFlags(
                Qt.TextSelectableByMouse
            )
            root.addWidget(self.lbl_compute, 0)
    
            self.lbl_compute.setSizePolicy(
                QSizePolicy.Expanding,
                QSizePolicy.Minimum,
            )
    
            self.btn_refresh.clicked.connect(
                self.refresh_compute
            )
    
        self.btn_copy.clicked.connect(self._on_copy)


    def _wire_store(self) -> None:
        """
        Optional auto-refresh when store changes.

        We keep this robust: if the store does not expose a
        signal, nothing breaks.
        """
        sig = getattr(self._store, "config_changed", None)
        if sig is None:
            sig = getattr(self._store, "changed", None)
        if sig is None:
            return

        try:
            sig.connect(self._on_store_changed)
        except Exception:
            return

    def _on_store_changed(self) -> None:
        # Compute can change when backend/device config changes.
        self.refresh_compute()

    # -------------------------------------------------
    # Public API
    # -------------------------------------------------
    def set_plan_text(self, text: str) -> None:
        self._plan_text = str(text or "")
        self.lbl_plan.setText(self._plan_text)

    def set_plan(
        self,
        *,
        epochs: int,
        warmup: int,
        ramp: int,
        mode: str,
        lambdas: Dict[str, float],
        scale: bool,
        eval_on: bool,
        future: bool,
    ) -> None:
        self.viz.set_plan(
            epochs=epochs,
            warmup=warmup,
            ramp=ramp,
            mode=mode,
            lambdas=lambdas,
            scale=scale,
            eval_on=eval_on,
            future=future,
        )

    def refresh_compute(self) -> None:
        if not hasattr(self, "lbl_compute"):
            return
        self.lbl_compute.setText(
            runtime_summary_text(self._store)
        )
    def refresh_all(self) -> None:
        if not self.lbl_plan.text():
            self.lbl_plan.setText(self._plan_text)
        self.refresh_compute()

    # -------------------------------------------------
    # Actions
    # -------------------------------------------------
    def _on_copy(self) -> None:
        txt = (self._plan_text or "").strip()
        if not txt:
            txt = (self.lbl_plan.text() or "").strip()
        if not txt:
            self.toast.emit("Nothing to copy.")
            return

        QApplication.clipboard().setText(txt)
        self.toast.emit("Run plan copied.")


class RunPreviewViz(QWidget):
    """
    Simple visual for:
    - warmup / ramp / steady timeline
    - lambda bars (λc, λgw, λp, λs, λmv)
    """

    def __init__(
        self,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setMinimumHeight(220)
        self.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Expanding,
        )

        self._epochs = 1
        self._warmup = 0
        self._ramp = 0
        self._mode = "both"
        self._scale = True
        self._eval = False
        self._future = False
        self._lmb: Dict[str, float] = {}

    def set_plan(
        self,
        *,
        epochs: int,
        warmup: int,
        ramp: int,
        mode: str,
        lambdas: Dict[str, float],
        scale: bool,
        eval_on: bool,
        future: bool,
    ) -> None:
        self._epochs = max(1, int(epochs))
        self._warmup = max(0, int(warmup))
        self._ramp = max(0, int(ramp))
        self._mode = str(mode)
        self._scale = bool(scale)
        self._eval = bool(eval_on)
        self._future = bool(future)
        self._lmb = dict(lambdas or {})
        self.update()

    def paintEvent(self, e: Any) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)

        r = self.rect().adjusted(8, 8, -8, -8)
        pal = self.palette()

        text_c = pal.windowText().color()
        base_c = pal.base().color()
        hi_c = pal.highlight().color()

        # background
        p.setPen(Qt.NoPen)
        bg = QColor(base_c)
        bg.setAlpha(180)
        p.setBrush(bg)
        p.drawRoundedRect(QRectF(r), 12, 12)

        # title
        p.setPen(QPen(text_c))
        p.drawText(
            r.adjusted(10, 8, -10, -8),
            Qt.AlignLeft | Qt.AlignTop,
            "Run plan visual",
        )

        # Timeline track
        tl = QRectF(
            r.left() + 10,
            r.top() + 34,
            r.width() - 20,
            22,
        )

        warm = int(self._warmup)
        ramp = int(self._ramp)
        steady = max(1, int((warm + ramp) * 0.6) or 12)
        total = max(1, warm + ramp + steady)

        w_w = tl.width() * (warm / total)
        w_r = tl.width() * (ramp / total)
        w_s = tl.width() * (steady / total)

        radius = 10.0

        path = QPainterPath()
        path.addRoundedRect(tl, radius, radius)

        track = QColor(text_c)
        track.setAlpha(25)

        p.setPen(Qt.NoPen)
        p.setBrush(track)
        p.drawPath(path)

        # clip fills inside rounded track
        p.save()
        p.setClipPath(path)

        c1 = QColor(text_c)
        c1.setAlpha(30)
        p.setBrush(c1)
        p.drawRect(
            QRectF(
                tl.left(),
                tl.top(),
                w_w,
                tl.height(),
            )
        )

        c2 = QColor(hi_c)
        c2.setAlpha(120)
        p.setBrush(c2)
        p.drawRect(
            QRectF(
                tl.left() + w_w,
                tl.top(),
                w_r,
                tl.height(),
            )
        )

        c3 = QColor(hi_c)
        c3.setAlpha(60)
        p.setBrush(c3)
        p.drawRect(
            QRectF(
                tl.left() + w_w + w_r,
                tl.top(),
                w_s,
                tl.height(),
            )
        )

        p.restore()

        p.setPen(QPen(text_c))
        p.drawText(
            QRectF(tl.left(), tl.bottom() + 6, tl.width(), 18),
            Qt.AlignLeft,
            f"warmup={warm}  ramp={ramp}  "
            f"epochs={self._epochs}",
        )

        # Lambda bars
        box = QRectF(
            r.left() + 10,
            tl.bottom() + 30,
            r.width() - 20,
            r.height() - (tl.bottom() - r.top()) - 40,
        )

        keys = ["λc", "λgw", "λp", "λs", "λmv"]
        vals = [
            float(self._lmb.get("c", 0.0)),
            float(self._lmb.get("gw", 0.0)),
            float(self._lmb.get("p", 0.0)),
            float(self._lmb.get("s", 0.0)),
            float(self._lmb.get("mv", 0.0)),
        ]
        vmax = max(1e-9, max(vals) if vals else 1.0)

        row_h = 18.0
        gap = 8.0
        y0 = box.top()

        for k, v in zip(keys, vals):
            p.setPen(QPen(text_c))
            p.drawText(
                QRectF(box.left(), y0, 44, row_h),
                Qt.AlignLeft | Qt.AlignVCenter,
                k,
            )

            bar = QRectF(
                box.left() + 44,
                y0 + 3,
                box.width() - 54,
                row_h - 6,
            )
            track2 = QColor(text_c)
            track2.setAlpha(25)
            p.setPen(Qt.NoPen)
            p.setBrush(track2)
            p.drawRoundedRect(bar, 6, 6)

            fill = QColor(hi_c)
            fill.setAlpha(140)
            p.setBrush(fill)
            wv = bar.width() * (v / vmax)
            p.drawRoundedRect(
                QRectF(bar.left(), bar.top(), wv, bar.height()),
                6,
                6,
            )

            p.setPen(QPen(text_c))
            p.drawText(
                QRectF(bar.right() - 60, y0, 60, row_h),
                Qt.AlignRight | Qt.AlignVCenter,
                f"{v:.2f}",
            )

            y0 += row_h + gap

        p.end()
