# -*- coding: utf-8 -*-
# License: BSD-3-Clause

from __future__ import annotations

import re

from PyQt5.QtCore import Qt, QRect, QSize
from PyQt5.QtGui import QColor, QFont, QFontMetrics, QPainter, QPalette
from PyQt5.QtWidgets import (
    QApplication,
    QStyle,
    QStyleOptionViewItem,
    QStyledItemDelegate,
)


_BADGE_RE = re.compile(r"^(.*)\s*\((\d+)\)\s*$")


def _rgba(hex_color: str, alpha: int) -> QColor:
    c = QColor(hex_color)
    c.setAlpha(int(alpha))
    return c


class NavCountBadgeDelegate(QStyledItemDelegate):
    """
    Draw a right-aligned badge pill for trailing '(n)'.

    This does NOT change the model/item text. It only
    changes how it's painted.
    """

    def __init__(
        self,
        parent=None,
        *,
        primary: str = "#2E3191",
        pad_x: int = 8,
        pad_y: int = 3,
        margin_r: int = 10,
    ):
        super().__init__(parent)
        self._primary = str(primary)
        self._pad_x = int(pad_x)
        self._pad_y = int(pad_y)
        self._margin_r = int(margin_r)

    def _split(self, text: str):
        m = _BADGE_RE.match(text or "")
        if not m:
            return text, None
        base = (m.group(1) or "").strip()
        cnt = int(m.group(2))
        return base, cnt

    def _badge_size(self, fm: QFontMetrics, badge: str):
        w = fm.horizontalAdvance(badge)
        h = fm.height()
        w = w + 2 * self._pad_x
        h = h + 2 * self._pad_y
        return w, h

    def paint(self, painter: QPainter, option, index):
        opt = QStyleOptionViewItem(option)
        self.initStyleOption(opt, index)

        base, cnt = self._split(opt.text)
        opt.text = ""

        widget = opt.widget
        style = widget.style() if widget else QApplication.style()

        # 1) Paint the standard item background/focus/etc
        style.drawControl(QStyle.CE_ItemViewItem, opt,
                          painter, widget)

        # 2) Compute text rect the "Qt way"
        text_opt = QStyleOptionViewItem(option)
        self.initStyleOption(text_opt, index)

        txt = base
        badge_w = 0
        badge_h = 0
        
        # Make parent/group items look like section headers
        if index.model().hasChildren(index):
            f = QFont(text_opt.font)
            f.setBold(True)
            text_opt.font = f
            
        fm = QFontMetrics(text_opt.font)

        if cnt is not None:
            badge = str(cnt)
            badge_w, badge_h = self._badge_size(fm, badge)

        text_rect = style.subElementRect(
            QStyle.SE_ItemViewItemText,
            text_opt,
            widget,
        )

        if cnt is not None:
            cut = badge_w + self._margin_r + 6
            text_rect = text_rect.adjusted(0, 0, -cut, 0)

        # 3) Paint the text (elided)
        painter.save()

        enabled = bool(text_opt.state & QStyle.State_Enabled)
        if enabled:
            pen = text_opt.palette.color(QPalette.Text)
        else:
            pen = text_opt.palette.color(QPalette.Disabled, QPalette.Text)
        
        painter.setPen(pen)
        
        el = fm.elidedText(txt, Qt.ElideRight,
                           max(0, text_rect.width()))
        painter.drawText(
            text_rect,
            Qt.AlignVCenter | Qt.AlignLeft,
            el,
        )

        painter.restore()

        # 4) Paint the badge pill (if any)
        if cnt is None:
            return

        badge = str(cnt)

        # Keep the pill height within the row
        row_h = opt.rect.height()
        max_h = max(16, row_h - 8)
        badge_h = min(badge_h, max_h)

        x = opt.rect.right() - badge_w - self._margin_r
        y = opt.rect.center().y() - (badge_h // 2)
        badge_rect = QRect(x, y, badge_w, badge_h)

        painter.save()
        painter.setRenderHint(QPainter.Antialiasing, True)

        sel = bool(opt.state & QStyle.State_Selected)
        hov = bool(opt.state & QStyle.State_MouseOver)

        if sel:
            bg = _rgba(self._primary, 255)
        elif hov:
            bg = _rgba(self._primary, 210)
        else:
            bg = _rgba(self._primary, 170)

        painter.setPen(Qt.NoPen)
        painter.setBrush(bg)

        r = badge_rect.height() // 2
        painter.drawRoundedRect(badge_rect, r, r)

        painter.setPen(QColor("white"))

        f = QFont(painter.font())
        f.setBold(True)
        painter.setFont(f)

        painter.drawText(
            badge_rect,
            Qt.AlignCenter,
            badge,
        )

        painter.restore()

    def sizeHint(self, option, index):
        sz = super().sizeHint(option, index)

        opt = QStyleOptionViewItem(option)
        self.initStyleOption(opt, index)

        _, cnt = self._split(opt.text)
        if cnt is None:
            return sz

        fm = QFontMetrics(opt.font)
        badge = str(cnt)

        w, _h = self._badge_size(fm, badge)
        extra = w + self._margin_r + 8

        return QSize(sz.width() + extra, sz.height())