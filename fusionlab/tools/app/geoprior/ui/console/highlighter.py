# geoprior/ui/console/highlighter.py
from __future__ import annotations

from PyQt5.QtGui import (
    QSyntaxHighlighter,
    QTextCharFormat,
    QFont,
)
from PyQt5.QtCore import QRegularExpression


class ConsoleHighlighter(QSyntaxHighlighter):
    def __init__(self, doc) -> None:
        super().__init__(doc)

        self._fmt_err = QTextCharFormat()
        self._fmt_err.setFontWeight(QFont.Bold)

        self._fmt_warn = QTextCharFormat()
        self._fmt_warn.setFontWeight(QFont.DemiBold)

        self._fmt_info = QTextCharFormat()

        self._re_err = QRegularExpression(r"\[ERROR\]")
        self._re_warn = QRegularExpression(r"\[WARN\]|\[WARNING\]")
        self._re_tb = QRegularExpression(r"\bTraceback\b")
        self._re_info = QRegularExpression(r"\[INFO\]")

    def highlightBlock(self, text: str) -> None:
        for re_, fmt in (
            (self._re_tb, self._fmt_err),
            (self._re_err, self._fmt_err),
            (self._re_warn, self._fmt_warn),
            (self._re_info, self._fmt_info),
        ):
            it = re_.globalMatch(text)
            while it.hasNext():
                m = it.next()
                self.setFormat(m.capturedStart(), m.capturedLength(), fmt)
