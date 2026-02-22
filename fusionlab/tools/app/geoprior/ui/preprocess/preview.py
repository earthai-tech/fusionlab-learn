

from __future__ import annotations

from typing import Any, List, Optional, Tuple

from PyQt5.QtCore import Qt, QRectF
from PyQt5.QtGui import (
    QColor,
    QPainter,
    QPen,
    QFont
)
from PyQt5.QtWidgets import (
    QWidget, 
    QSizePolicy,  
    QAbstractItemView,
    QHeaderView,
    QTableWidget,
    QTableWidgetItem,
)

class Stage1PreviewViz(QWidget):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(160)
        self.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Fixed,
        )

        self._inputs_ok = False
        self._has_best = False
        self._complete = False
        self._match = False
        self._decision = "WAITING"

        self._auto = True
        self._force = True
        self._clean = False
        self._future = False

    def set_state(
        self,
        *,
        inputs_ok: bool,
        has_best: bool,
        complete: bool,
        match: bool,
        decision: str,
        auto_reuse: bool,
        force_rb: bool,
        clean: bool,
        future: bool,
    ) -> None:
        self._inputs_ok = bool(inputs_ok)
        self._has_best = bool(has_best)
        self._complete = bool(complete)
        self._match = bool(match)
        self._decision = str(decision or "WAITING")

        self._auto = bool(auto_reuse)
        self._force = bool(force_rb)
        self._clean = bool(clean)
        self._future = bool(future)

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
            "Stage-1 overview",
        )

        # -------------------------
        # Track: Inputs | Best | Action
        # -------------------------
        tl = QRectF(
            r.left() + 10,
            r.top() + 34,
            r.width() - 20,
            20,
        )

        track = QColor(text_c)
        track.setAlpha(25)
        p.setPen(Qt.NoPen)
        p.setBrush(track)
        p.drawRoundedRect(tl, 10, 10)

        seg_w = tl.width() / 3.0
        s1 = QRectF(tl.left(), tl.top(), seg_w, tl.height())
        s2 = QRectF(
            tl.left() + seg_w,
            tl.top(),
            seg_w,
            tl.height(),
        )
        s3 = QRectF(
            tl.left() + 2.0 * seg_w,
            tl.top(),
            seg_w,
            tl.height(),
        )

        def fill_seg(rr: QRectF, alpha: int) -> None:
            c = QColor(hi_c)
            c.setAlpha(alpha)
            p.setBrush(c)
            p.drawRoundedRect(rr, 10, 10)

        # Inputs
        fill_seg(s1, 120 if self._inputs_ok else 0)

        # Best run
        if self._has_best:
            if self._complete and self._match:
                fill_seg(s2, 140)
            else:
                fill_seg(s2, 80)
        else:
            fill_seg(s2, 0)

        # Action
        d = (self._decision or "").strip()
        if d == "REUSE":
            fill_seg(s3, 160)
        elif d in ("BUILD", "REBUILD"):
            fill_seg(s3, 100)
        elif d.startswith("WAIT"):
            fill_seg(s3, 0)
        else:
            fill_seg(s3, 70)

        p.setPen(QPen(text_c))
        p.drawText(
            QRectF(tl.left(), tl.bottom() + 6, seg_w, 16),
            Qt.AlignLeft,
            "Inputs",
        )
        p.drawText(
            QRectF(s2.left(), tl.bottom() + 6, seg_w, 16),
            Qt.AlignLeft,
            "Best run",
        )
        p.drawText(
            QRectF(s3.left(), tl.bottom() + 6, seg_w, 16),
            Qt.AlignLeft,
            "Action",
        )

        # -------------------------
        # Option bars (ON/OFF)
        # -------------------------
        box = QRectF(
            r.left() + 10,
            tl.bottom() + 30,
            r.width() - 20,
            r.height() - (tl.bottom() - r.top()) - 40,
        )

        opts = [
            ("Auto-reuse", self._auto),
            ("Force rebuild", self._force),
            ("Clean", self._clean),
            ("Future NPZ", self._future),
        ]

        row_h = 16.0
        gap = 8.0
        y0 = box.top()

        for name, on in opts:
            p.setPen(QPen(text_c))
            p.drawText(
                QRectF(box.left(), y0, 90, row_h),
                Qt.AlignLeft | Qt.AlignVCenter,
                name,
            )

            bar = QRectF(
                box.left() + 90,
                y0 + 3,
                box.width() - 100,
                row_h - 6,
            )

            t2 = QColor(text_c)
            t2.setAlpha(25)
            p.setPen(Qt.NoPen)
            p.setBrush(t2)
            p.drawRoundedRect(bar, 6, 6)

            if on:
                f2 = QColor(hi_c)
                f2.setAlpha(140)
                p.setBrush(f2)
                p.drawRoundedRect(bar, 6, 6)

            y0 += row_h + gap

        p.end()

class RecapTable(QTableWidget):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(0, 2, parent)

        self.setObjectName("recapTable")
        self.setShowGrid(False)
        self.setAlternatingRowColors(True)
        self.setEditTriggers(QTableWidget.NoEditTriggers)

        self.setSelectionMode(QAbstractItemView.NoSelection)
        self.setFocusPolicy(Qt.NoFocus)
        self.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Fixed,
        )
        self.setVerticalScrollBarPolicy(
            Qt.ScrollBarAlwaysOff
        )
        self.setHorizontalScrollBarPolicy(
            Qt.ScrollBarAlwaysOff
        )
        
        self.horizontalHeader().hide()
        self.verticalHeader().hide()

        hh = self.horizontalHeader()
        hh.setSectionResizeMode(
            0, QHeaderView.ResizeToContents
        )
        hh.setSectionResizeMode(1, QHeaderView.Stretch)

        self._bold = QFont()
        self._bold.setBold(True)

    def _fit_height(self) -> None:
        h = 6 + self.frameWidth() * 2
        for i in range(self.rowCount()):
            h += self.rowHeight(i)
        self.setFixedHeight(max(70, h))
        
    def set_rows(self, rows: List[Tuple[str, str]]) -> None:
        rows = list(rows or [])
        self.setRowCount(len(rows))

        for i, (k, v) in enumerate(rows):
            it_k = QTableWidgetItem(str(k))
            it_k.setFont(self._bold)

            it_v = QTableWidgetItem(str(v))
            it_v.setToolTip(str(v))

            self.setItem(i, 0, it_k)
            self.setItem(i, 1, it_v)

        self.resizeRowsToContents()
        self._fit_height()