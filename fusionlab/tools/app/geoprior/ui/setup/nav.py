# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>

"""geoprior.ui.setup.nav

Modern left navigator for Setup sections.

Features
--------
- Two-line items: title + optional description.
- Hover slide animation (subtle x-shift).
- Selected accent bar on the left.
- Filter support (external or internal search).
- Emits selected section id; parent scrolls to the card.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple

from PyQt5.QtCore import (
    QEasingCurve,
    Qt,
    QVariantAnimation,
    pyqtSignal,
    QRect
)
from PyQt5.QtGui import ( 
    QFont, 
    QPainter, 
    QColor, 
    QFontMetrics,
    # QPalette, 
    # QPen
)
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QStyledItemDelegate,
    QStyle,
    QStyleOptionViewItem,
    QVBoxLayout,
    QWidget,
)
from ...styles import SECONDARY, SECONDARY_TBLUE

ROLE_ID = int(Qt.UserRole)
ROLE_DESC = int(Qt.UserRole) + 1


Section = Tuple[str, str]


@dataclass(frozen=True)
class NavSection:
    sec_id: str
    title: str
    description: str = ""


def _to_nav_sections(
    sections: Sequence[Any],
) -> List[NavSection]:
    out: List[NavSection] = []
    for s in (sections or []):
        if isinstance(s, NavSection):
            out.append(s)
            continue

        # Accept (id, title) or (id, title, desc)
        if isinstance(s, (tuple, list)):
            if len(s) >= 2:
                sec_id = str(s[0])
                title = str(s[1])
                desc = ""
                if len(s) >= 3 and s[2] is not None:
                    desc = str(s[2])
                out.append(
                    NavSection(
                        sec_id=sec_id,
                        title=title,
                        description=desc,
                    )
                )
            continue

        # Accept objects with attrs: sec_id/title/description
        sec_id = getattr(s, "sec_id", None)
        title = getattr(s, "title", None)
        if sec_id is None or title is None:
            continue

        desc = getattr(s, "description", "") or ""
        out.append(
            NavSection(
                sec_id=str(sec_id),
                title=str(title),
                description=str(desc),
            )
        )

    return out

class _NavDelegate(QStyledItemDelegate):
    """Paint 2-line items with hover + selected accent."""

    def __init__(self, view: _NavList) -> None:
        super().__init__(view)
        self._view = view

    def paint(
        self,
        painter: QPainter,
        option: QStyleOptionViewItem,
        index,
    ) -> None:
        painter.save()
        try:
            row = int(index.row())
            off = float(self._view.hover_offset(row))
    
            opt = QStyleOptionViewItem(option)
            self.initStyleOption(opt, index)
    
            # Slight inset, no rounded shapes
            rect = opt.rect.adjusted(8, 4, -8, -4)
            rect = rect.adjusted(int(off), 0, int(off), 0)
    
            pal = opt.palette
            selected = bool(opt.state & QStyle.State_Selected)
            hovered = bool(opt.state & QStyle.State_MouseOver)
    
            win = pal.window().color()
            is_dark = bool(win.value() < 128)
    
            painter.setRenderHint(QPainter.Antialiasing, True)
    
            # ------------------------------------------------------------
            # Backgrounds: flat fill only (no border, no rounding)
            # ------------------------------------------------------------
            if hovered and not selected:
                bg = QColor(SECONDARY_TBLUE)
                bg.setAlpha(26 if is_dark else 18)  # subtle
                painter.fillRect(rect, bg)
    
            if selected:
                bg = QColor(SECONDARY)
                bg.setAlpha(40 if is_dark else 22)  # subtle but visible
                painter.fillRect(rect, bg)
    
                # Accent bar (LEFT) - keep it strong, clean
                bar = QRect(rect.left(), rect.top() + 2, 4, rect.height() - 4)
                painter.fillRect(bar, QColor(SECONDARY))
    
                # Optional: thin right hairline for extra “selected” clarity
                # hair = QRect(rect.right() - 1, rect.top() + 2, 1, rect.height() - 4)
                # c = QColor(SECONDARY); c.setAlpha(170 if is_dark else 140)
                # painter.fillRect(hair, c)
    
            # ------------------------------------------------------------
            # Text layout + eliding (prevents hiding / clipping)
            # ------------------------------------------------------------
            title = str(index.data(Qt.DisplayRole) or "")
            desc = str(index.data(ROLE_DESC) or "")
    
            x0 = rect.left() + 12
            w = rect.width() - 16
    
            f1: QFont = opt.font
            f1.setBold(True)
    
            f2: QFont = opt.font
            f2.setPointSize(max(8, f2.pointSize() - 1))
    
            fm1 = QFontMetrics(f1)
            fm2 = QFontMetrics(f2)
    
            th = max(18, fm1.height())
            dh = max(16, fm2.height())
            gap = 2
    
            has_desc = bool(desc.strip())
            block_h = th if not has_desc else (th + gap + dh)
            top = rect.center().y() - (block_h // 2)
    
            title_rect = QRect(x0, top, w, th)
            desc_rect = QRect(x0, top + th + gap, w, dh)
    
            # Elide to avoid truncation / hidden text
            title_draw = fm1.elidedText(title, Qt.ElideRight, title_rect.width())
            desc_draw = fm2.elidedText(desc, Qt.ElideRight, desc_rect.width())
    
            # Title color rules (never white-on-light)
            if selected:
                # Orange title (modern “selected” signal)
                painter.setPen(QColor(SECONDARY))
            else:
                painter.setPen(pal.text().color())
    
            painter.setFont(f1)
            painter.drawText(title_rect, Qt.AlignVCenter | Qt.AlignLeft, title_draw)
    
            if has_desc:
                c2 = pal.text().color()
                c2.setAlpha(190 if is_dark else 160)
                painter.setPen(c2)
                painter.setFont(f2)
                painter.drawText(desc_rect, Qt.AlignVCenter | Qt.AlignLeft, desc_draw)
    
        finally:
            painter.restore()


    def sizeHint(self, option, index):  # type: ignore[override]
        base = super().sizeHint(option, index)
        desc = str(index.data(ROLE_DESC) or "")
        h = 52 if desc.strip() else 40
        base.setHeight(max(base.height(), h))
        return base


class _NavList(QListWidget):
    """List widget that tracks hover row + anim offsets."""

    def __init__(
        self,
        parent: Optional[QWidget] = None,
        *,
        hover_shift: int = 7,
    ) -> None:
        super().__init__(parent)
        self._hover_shift = int(max(0, hover_shift))
        self._hover_row = -1
        self._offsets: dict[int, float] = {}
        self._anims: dict[int, QVariantAnimation] = {}

        self.setMouseTracking(True)
        self.setUniformItemSizes(False)

        self.setSelectionMode(
            QAbstractItemView.SingleSelection
        )

    def hover_offset(self, row: int) -> float:
        return float(self._offsets.get(int(row), 0.0))

    def _set_hover_row(self, row: int) -> None:
        row = int(row)
        if row == self._hover_row:
            return

        prev = int(self._hover_row)
        self._hover_row = row

        if prev >= 0:
            self._animate_row(prev, 0.0)

        if row >= 0:
            self._animate_row(row, float(self._hover_shift))

    def _animate_row(self, row: int, end: float) -> None:
        row = int(row)
        start = float(self._offsets.get(row, 0.0))

        if abs(start - float(end)) < 0.1:
            self._offsets[row] = float(end)
            self.viewport().update()
            return

        old = self._anims.get(row)
        if old is not None:
            try:
                old.stop()
            except Exception:
                pass

        anim = QVariantAnimation(self)
        anim.setStartValue(start)
        anim.setEndValue(float(end))
        anim.setDuration(140)
        anim.setEasingCurve(QEasingCurve.OutCubic)

        def _on_val(v: Any) -> None:
            try:
                self._offsets[row] = float(v)
            except Exception:
                self._offsets[row] = float(end)
            self.viewport().update()

        def _on_done() -> None:
            self._offsets[row] = float(end)
            self._anims.pop(row, None)
            self.viewport().update()

        anim.valueChanged.connect(_on_val)
        anim.finished.connect(_on_done)

        self._anims[row] = anim
        anim.start()

    def mouseMoveEvent(self, ev) -> None:  # type: ignore[override]
        idx = self.indexAt(ev.pos())
        self._set_hover_row(int(idx.row()) if idx.isValid() else -1)
        super().mouseMoveEvent(ev)

    def leaveEvent(self, ev) -> None:  # type: ignore[override]
        self._set_hover_row(-1)
        super().leaveEvent(ev)


class SetupNav(QWidget):
    """Left section navigator (modern UX)."""

    section_changed = pyqtSignal(str)
    filter_changed = pyqtSignal(str)

    def __init__(
        self,
        parent: Optional[QWidget] = None,
        *,
        with_search: bool = True,
        show_descriptions: bool = True,
        hover_shift: int = 4,
    ) -> None:
        super().__init__(parent)
        self._show_desc = bool(show_descriptions)
        self._sections: List[NavSection] = []

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(8)

        # Header row (title + count)
        head = QWidget(self)
        head_l = QHBoxLayout(head)
        head_l.setContentsMargins(8, 6, 8, 0)
        head_l.setSpacing(8)

        self.lbl_title = QLabel("Sections", head)
        self.lbl_count = QLabel("", head)
        self.lbl_count.setObjectName("setupNavCount")

        head_l.addWidget(self.lbl_title)
        head_l.addStretch(1)
        head_l.addWidget(self.lbl_count)
        lay.addWidget(head, 0)

        # Optional local search
        self.search: Optional[QLineEdit] = None
        if with_search:
            self.search = QLineEdit(self)
            self.search.setClearButtonEnabled(True)
            self.search.setPlaceholderText(
                "Filter sections…"
            )
            lay.addWidget(self.search, 0)

        # List
        self.list = _NavList(self, hover_shift=hover_shift)
        self.list.setObjectName("setupNavList")
        self.list.setAlternatingRowColors(False)
        self.list.setVerticalScrollMode(
            QAbstractItemView.ScrollPerPixel
        )
        self.list.setHorizontalScrollBarPolicy(
            Qt.ScrollBarAlwaysOff
        )

        self.list.setItemDelegate(_NavDelegate(self.list))
        self.list.currentItemChanged.connect(
            self._on_current_changed
        )
        lay.addWidget(self.list, 1)

        # Styling (safe defaults; can be overridden globally)
        self.setObjectName("setupNav")
        self.setStyleSheet(
            "\n".join(
                [
                    "QWidget#setupNav {",
                    "  background: transparent;",
                    "}",
        
                    "QLabel#setupNavCount {",
                    "  opacity: 0.85;",
                    "}",
        
                    "QListWidget#setupNavList {",
                    "  border: 1px solid rgba(0,0,0,0.10);",
                    "  border-radius: 12px;",
                    "  padding: 6px;",
                    "  outline: 0;",
                    "  background: rgba(255,255,255,0.55);",
                    "}",
        
                    "QListWidget#setupNavList::item {",
                    "  margin: 2px 0px;",
                    "  border-radius: 0px;",
                    "}",
        
                    "QListWidget#setupNavList::item:selected {",
                    "  background: transparent;",
                    "}",
                    "QListWidget#setupNavList::item:hover {",
                    "  background: transparent;",
                    "}",
                ]
            )
        )


        if self.search is not None:
            self.search.textChanged.connect(self._on_filter_text)

    # ------------------------------------------------------------------
    # Population / filtering
    # ------------------------------------------------------------------
    def set_sections(self, sections: Sequence[Any]) -> None:
        """Replace list content.

        Accepts:
        - (sec_id, title)
        - (sec_id, title, description)
        - objects with sec_id/title/description attrs
        - NavSection / SectionSpec-like items
        """
        self._sections = _to_nav_sections(sections)

        self.list.clear()
        for s in self._sections:
            it = QListWidgetItem(str(s.title), self.list)
            it.setData(ROLE_ID, str(s.sec_id))

            desc = str(s.description or "")
            if not self._show_desc:
                desc = ""

            it.setData(ROLE_DESC, desc)

        self._update_count()

        if self.list.count() > 0:
            self.list.setCurrentRow(0)

    def apply_filter(self, text: str) -> None:
        """Hide items that do not match the query."""
        q = (text or "").strip().lower()

        for i in range(self.list.count()):
            it = self.list.item(i)
            title = (it.text() or "").lower()
            desc = str(it.data(ROLE_DESC) or "").lower()

            hit = (not q) or (q in title) or (q in desc)
            it.setHidden(not bool(hit))

        self._update_count()

        cur = self.list.currentItem()
        if cur is not None and cur.isHidden():
            self.select_first_visible()

    def select_first_visible(self) -> None:
        for i in range(self.list.count()):
            it = self.list.item(i)
            if not it.isHidden():
                self.list.setCurrentItem(it)
                return

    def current_section_id(self) -> Optional[str]:
        cur = self.list.currentItem()
        if cur is None:
            return None
        v = cur.data(ROLE_ID)
        return None if v is None else str(v)

    def set_current_section_id(self, sec_id: str) -> None:
        want = str(sec_id)
        for i in range(self.list.count()):
            it = self.list.item(i)
            if str(it.data(ROLE_ID) or "") == want:
                if not it.isHidden():
                    self.list.setCurrentItem(it)
                return

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _update_count(self) -> None:
        vis = 0
        for i in range(self.list.count()):
            if not self.list.item(i).isHidden():
                vis += 1

        total = int(self.list.count())
        self.lbl_count.setText(f"{vis}/{total}")

    def _on_filter_text(self, text: str) -> None:
        self.apply_filter(text)
        self.filter_changed.emit(str(text or ""))

    def _on_current_changed(
        self,
        cur: Optional[QListWidgetItem],
        _prev: Optional[QListWidgetItem],
    ) -> None:
        if cur is None:
            return
        sec_id = cur.data(ROLE_ID)
        if sec_id is None:
            return
        self.section_changed.emit(str(sec_id))
